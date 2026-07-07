use clap::ValueEnum;
use reqwest::header::HeaderMap;
use serde::Serialize;
use std::{
    fs::File,
    io::{self, BufRead, Read},
    path::Path,
    process::Command,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use uuid::Uuid;

use crate::Info;

const TELEMETRY_URL: &str = "https://huggingface.co/api/telemetry/tei";

#[derive(Copy, Clone, Debug, Serialize, ValueEnum)]
pub enum UsageStatsLevel {
    On,
    Off,
    NoStack,
}

#[derive(Debug, Clone, Serialize)]
pub struct UserAgent {
    pub uid: String,
    pub args: Args,
    pub env: Env,
}

impl UserAgent {
    pub fn new(args: Args) -> Self {
        Self {
            uid: Uuid::new_v4().to_string(),
            args,
            env: Env::new(),
        }
    }
}

#[derive(Serialize, Debug)]
pub enum EventType {
    Start,
    Stop,
    Error,
    Ping,
}

#[derive(Debug, Serialize)]
pub struct UsageStatsEvent {
    user_agent: UserAgent,
    event_type: EventType,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_reason: Option<String>,
}

impl UsageStatsEvent {
    pub fn new(user_agent: UserAgent, event_type: EventType, error_reason: Option<String>) -> Self {
        Self {
            user_agent,
            event_type,
            error_reason,
        }
    }

    pub async fn send(&self) {
        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse().unwrap());

        let Ok(body) = serde_json::to_string(&self) else {
            return;
        };

        let client = reqwest::Client::new();
        let _ = client
            .post(TELEMETRY_URL)
            .headers(headers)
            .body(body)
            .timeout(Duration::from_secs(10))
            .send()
            .await;
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Args {
    info: Info,
    usage_stats_level: UsageStatsLevel,
    origin: Option<String>,
}

impl Args {
    pub fn new(info: Info, usage_stats_level: UsageStatsLevel, origin: Option<String>) -> Self {
        Self {
            info,
            usage_stats_level,
            origin,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Env {
    git_sha: &'static str,
    docker_label: &'static str,
    system_env: SystemInfo,
    hardware: HardwareInfo,
    instance_type: Option<String>,
}

impl Env {
    pub fn new() -> Self {
        Self {
            system_env: SystemInfo::new(),
            hardware: HardwareInfo::new(),
            instance_type: instance_type(),
            git_sha: option_env!("VERGEN_GIT_SHA").unwrap_or("N/A"),
            docker_label: option_env!("DOCKER_LABEL").unwrap_or("N/A"),
        }
    }
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemInfo {
    cpu_count: usize,
    total_memory: u64,
    architecture: String,
    platform: String,
}

impl SystemInfo {
    fn new() -> Self {
        Self {
            cpu_count: num_cpus::get(),
            total_memory: total_memory(),
            architecture: std::env::consts::ARCH.to_string(),
            platform: format!(
                "{}-{}-{}",
                std::env::consts::OS,
                std::env::consts::FAMILY,
                std::env::consts::ARCH
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct HardwareInfo {
    devices: Vec<DeviceInfo>,
}

impl HardwareInfo {
    fn new() -> Self {
        let mut devices = Vec::new();
        devices.extend(nvidia_devices());
        devices.extend(amd_devices());
        devices.extend(intel_xpu_devices());
        devices.extend(intel_hpu_devices());
        devices.extend(apple_mps_devices());

        if devices.is_empty() {
            devices.push(DeviceInfo::cpu());
        }

        Self { devices }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    kind: &'static str,
    vendor: &'static str,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    driver_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_total: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_cap: Option<String>,
}

impl DeviceInfo {
    fn cpu() -> Self {
        Self {
            kind: "cpu",
            vendor: "generic",
            name: cpu_model().unwrap_or_else(|| "CPU".to_string()),
            driver_version: None,
            memory_total: None,
            compute_cap: None,
        }
    }
}

pub fn is_container() -> io::Result<bool> {
    let path = Path::new("/proc/self/cgroup");
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if line.contains("/docker/")
            || line.contains("/docker-")
            || line.contains("/kubepods/")
            || line.contains("/kubepods-")
            || line.contains("containerd")
            || line.contains("crio")
            || line.contains("podman")
        {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn user_agent(info: Info, usage_stats_level: UsageStatsLevel) -> Option<UserAgent> {
    let is_container = matches!(is_container(), Ok(true));
    match (usage_stats_level, is_container) {
        (UsageStatsLevel::On | UsageStatsLevel::NoStack, true) => {
            let origin = std::env::var("HF_HUB_USER_AGENT_ORIGIN").ok();
            Some(UserAgent::new(Args::new(info, usage_stats_level, origin)))
        }
        _ => None,
    }
}

pub fn spawn_ping_task(user_agent: UserAgent) -> Arc<AtomicBool> {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

    tokio::spawn(async move {
        UsageStatsEvent::new(user_agent.clone(), EventType::Start, None)
            .send()
            .await;

        while !stop_clone.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_secs(900)).await;
            if stop_clone.load(Ordering::Relaxed) {
                break;
            }

            UsageStatsEvent::new(user_agent.clone(), EventType::Ping, None)
                .send()
                .await;
        }
    });

    stop
}

fn total_memory() -> u64 {
    let mut meminfo = String::new();
    let Ok(mut file) = File::open("/proc/meminfo") else {
        return 0;
    };
    if file.read_to_string(&mut meminfo).is_err() {
        return 0;
    }

    meminfo
        .lines()
        .find_map(|line| {
            let value = line.strip_prefix("MemTotal:")?.trim();
            let kb = value.split_whitespace().next()?.parse::<u64>().ok()?;
            Some(kb * 1024)
        })
        .unwrap_or(0)
}

fn instance_type() -> Option<String> {
    ["TEI_INSTANCE_TYPE", "INSTANCE_TYPE", "CLOUD_INSTANCE_TYPE"]
        .into_iter()
        .find_map(|key| std::env::var(key).ok().filter(|value| !value.is_empty()))
}

fn command_output(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }

    String::from_utf8(output.stdout)
        .ok()
        .map(|stdout| stdout.trim().to_string())
        .filter(|stdout| !stdout.is_empty())
}

fn nvidia_devices() -> Vec<DeviceInfo> {
    let Some(output) = command_output(
        "nvidia-smi",
        &[
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
    ) else {
        return Vec::new();
    };

    output
        .lines()
        .filter_map(|line| {
            let fields = line
                .split(',')
                .map(|field| field.trim().to_string())
                .collect::<Vec<_>>();
            let name = fields.first()?.to_string();
            Some(DeviceInfo {
                kind: "gpu",
                vendor: "nvidia",
                name,
                driver_version: fields.get(1).cloned().filter(|field| !field.is_empty()),
                memory_total: fields
                    .get(2)
                    .filter(|field| !field.is_empty())
                    .map(|field| format!("{field} MiB")),
                compute_cap: fields.get(3).cloned().filter(|field| !field.is_empty()),
            })
        })
        .collect()
}

fn amd_devices() -> Vec<DeviceInfo> {
    rocm_smi_devices()
        .or_else(amd_smi_devices)
        .unwrap_or_default()
}

fn amd_smi_devices() -> Option<Vec<DeviceInfo>> {
    let output = command_output("amd-smi", &["static", "--gpu", "all", "--json"])?;
    let json: serde_json::Value = serde_json::from_str(&output).ok()?;
    let mut devices = Vec::new();
    collect_amd_json_devices(&json, &mut devices);

    if devices.is_empty() {
        None
    } else {
        Some(devices)
    }
}

fn collect_amd_json_devices(value: &serde_json::Value, devices: &mut Vec<DeviceInfo>) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                collect_amd_json_devices(value, devices);
            }
        }
        serde_json::Value::Object(map) => {
            let name = map
                .get("market_name")
                .or_else(|| map.get("product_name"))
                .or_else(|| map.get("device_name"))
                .and_then(serde_json::Value::as_str);

            if let Some(name) = name {
                let driver_version = map
                    .get("driver_version")
                    .or_else(|| map.get("version"))
                    .and_then(serde_json::Value::as_str)
                    .map(ToString::to_string);
                let memory_total = map
                    .get("vram_size")
                    .or_else(|| map.get("memory_total"))
                    .and_then(|value| match value {
                        serde_json::Value::String(value) => Some(value.to_string()),
                        serde_json::Value::Number(value) => Some(value.to_string()),
                        _ => None,
                    });

                devices.push(DeviceInfo {
                    kind: "gpu",
                    vendor: "amd",
                    name: name.to_string(),
                    driver_version,
                    memory_total,
                    compute_cap: None,
                });
            } else {
                for value in map.values() {
                    collect_amd_json_devices(value, devices);
                }
            }
        }
        _ => {}
    }
}

fn rocm_smi_devices() -> Option<Vec<DeviceInfo>> {
    let output = command_output("rocm-smi", &["--showproductname", "--showmeminfo", "vram"])?;
    let mut devices = Vec::new();

    for line in output.lines() {
        let Some((prefix, value)) = line.split_once(':') else {
            continue;
        };
        if !prefix.contains("GPU") || !prefix.to_lowercase().contains("card series") {
            continue;
        }
        devices.push(DeviceInfo {
            kind: "gpu",
            vendor: "amd",
            name: value.trim().to_string(),
            driver_version: None,
            memory_total: None,
            compute_cap: None,
        });
    }

    if devices.is_empty() {
        None
    } else {
        Some(devices)
    }
}

fn intel_xpu_devices() -> Vec<DeviceInfo> {
    let Some(output) = command_output("xpu-smi", &["discovery", "-j"]) else {
        return Vec::new();
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&output) else {
        return Vec::new();
    };

    let mut devices = Vec::new();
    collect_intel_xpu_json_devices(&json, &mut devices);
    devices
}

fn collect_intel_xpu_json_devices(value: &serde_json::Value, devices: &mut Vec<DeviceInfo>) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                collect_intel_xpu_json_devices(value, devices);
            }
        }
        serde_json::Value::Object(map) => {
            let name = map
                .get("Device Name")
                .or_else(|| map.get("device_name"))
                .or_else(|| map.get("name"))
                .and_then(serde_json::Value::as_str);

            if let Some(name) = name {
                devices.push(DeviceInfo {
                    kind: "gpu",
                    vendor: "intel",
                    name: name.to_string(),
                    driver_version: None,
                    memory_total: None,
                    compute_cap: None,
                });
            } else {
                for value in map.values() {
                    collect_intel_xpu_json_devices(value, devices);
                }
            }
        }
        _ => {}
    }
}

fn intel_hpu_devices() -> Vec<DeviceInfo> {
    let Some(output) = command_output(
        "hl-smi",
        &[
            "--query-aip=name,driver_version,memory.total",
            "--format=csv",
        ],
    ) else {
        return Vec::new();
    };

    output
        .lines()
        .skip(1)
        .filter_map(|line| {
            let fields = line
                .split(',')
                .map(|field| field.trim().to_string())
                .collect::<Vec<_>>();
            let name = fields.first()?.to_string();
            Some(DeviceInfo {
                kind: "hpu",
                vendor: "intel",
                name,
                driver_version: fields.get(1).cloned().filter(|field| !field.is_empty()),
                memory_total: fields.get(2).cloned().filter(|field| !field.is_empty()),
                compute_cap: None,
            })
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn apple_mps_devices() -> Vec<DeviceInfo> {
    let Some(output) = command_output("system_profiler", &["SPDisplaysDataType"]) else {
        return Vec::new();
    };

    output
        .lines()
        .filter_map(|line| {
            let name = line.trim().strip_prefix("Chipset Model:")?.trim();
            Some(DeviceInfo {
                kind: "mps",
                vendor: "apple",
                name: name.to_string(),
                driver_version: None,
                memory_total: None,
                compute_cap: None,
            })
        })
        .collect()
}

#[cfg(not(target_os = "macos"))]
fn apple_mps_devices() -> Vec<DeviceInfo> {
    Vec::new()
}

#[cfg(target_os = "linux")]
fn cpu_model() -> Option<String> {
    let mut cpuinfo = String::new();
    File::open("/proc/cpuinfo")
        .ok()?
        .read_to_string(&mut cpuinfo)
        .ok()?;

    cpuinfo.lines().find_map(|line| {
        let value = line.strip_prefix("model name")?.split_once(':')?.1.trim();
        Some(value.to_string())
    })
}

#[cfg(target_os = "macos")]
fn cpu_model() -> Option<String> {
    command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn cpu_model() -> Option<String> {
    None
}
