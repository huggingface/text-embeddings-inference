use crate::logging::log_lines;
use std::ffi::OsString;
use std::io::{BufRead, BufReader};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread::sleep;
use std::time::{Duration, Instant};
use std::{env, fs, io, thread};
use text_embeddings_backend_core::{BackendError, Pool};

#[derive(Debug)]
pub(crate) struct BackendProcess {
    inner: Child,
}

impl BackendProcess {
    pub(crate) fn new(
        model_path: String,
        dtype: String,
        uds_path: &str,
        otlp_endpoint: Option<String>,
        otlp_service_name: String,
        pool: Pool,
    ) -> Result<Self, BackendError> {
        // Get UDS path
        let uds = Path::new(uds_path);

        // Clean previous runs
        if uds.exists() {
            fs::remove_file(uds).expect("could not remove UDS file");
        }

        let pool = match pool {
            Pool::Cls => "cls",
            Pool::Mean => "mean",
            Pool::LastToken => "lasttoken",
            Pool::Splade => {
                return Err(BackendError::Start(format!("{pool:?} is not supported")));
            }
        };

        // Process args
        let mut python_server_args = vec![
            model_path,
            "--dtype".to_owned(),
            dtype,
            "--uds-path".to_owned(),
            uds_path.to_owned(),
            "--logger-level".to_owned(),
            "INFO".to_owned(),
            "--json-output".to_owned(),
            "--pool".to_owned(),
            pool.to_owned(),
        ];

        // OpenTelemetry
        if let Some(otlp_endpoint) = otlp_endpoint {
            python_server_args.push("--otlp-endpoint".to_owned());
            python_server_args.push(otlp_endpoint);
        }

        python_server_args.push("--otlp-service-name".to_owned());
        python_server_args.push(otlp_service_name);

        // Copy current process env
        let envs: Vec<(OsString, OsString)> = env::vars_os().collect();

        tracing::info!("Starting Python backend");
        let mut p = match Command::new("python-text-embeddings-server")
            .args(python_server_args)
            .envs(envs)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .spawn()
        {
            Ok(p) => p,
            Err(err) => {
                if err.kind() == io::ErrorKind::NotFound {
                    return Err(BackendError::Start(
                        "python-text-embeddings-server not found in PATH".to_owned(),
                    ));
                }
                return Err(BackendError::Start(err.to_string()));
            }
        };

        let stdout_reader = BufReader::new(p.stdout.take().unwrap());
        let stderr_reader = BufReader::new(p.stderr.take().unwrap());

        //stdout tracing thread
        thread::spawn(move || {
            let _span = tracing::span!(tracing::Level::INFO, "python-backend").entered();
            log_lines(stdout_reader.lines());
        });

        let start_time = Instant::now();
        let mut wait_time = Instant::now();

        loop {
            // Process exited
            if let Some(exit_status) = p.try_wait().unwrap() {
                // We read stderr in another thread as it seems that lines() can block in some cases
                let (err_sender, err_receiver) = mpsc::channel();
                thread::spawn(move || {
                    for line in stderr_reader.lines().map_while(Result::ok) {
                        err_sender.send(line).unwrap_or(());
                    }
                });
                let mut err = String::new();
                while let Ok(line) = err_receiver.recv_timeout(Duration::from_millis(10)) {
                    err = err + "\n" + &line;
                }

                tracing::debug!("Python Backend complete standard error output:\n{err}");

                if let Some(signal) = exit_status.signal() {
                    return Err(BackendError::Start(format!(
                        "Python Backend process was signaled to shutdown with signal {signal}"
                    )));
                }
                return Err(BackendError::Start(
                    "Python backend failed to start".to_string(),
                ));
            }

            // Shard is ready
            if uds.exists() {
                tracing::info!("Python backend ready in {:?}", start_time.elapsed());
                break;
            } else if wait_time.elapsed() > Duration::from_secs(10) {
                tracing::info!("Waiting for Python backend to be ready...");
                wait_time = Instant::now();
            }
            sleep(Duration::from_millis(5));
        }

        Ok(Self { inner: p })
    }
}

impl Drop for BackendProcess {
    fn drop(&mut self) {
        self.inner.kill().unwrap();
        let _ = self.inner.wait();
        tracing::info!("Python backend process terminated");
    }
}
