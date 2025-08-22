use serde::Deserialize;
use std::io::{BufRead, Lines};

#[derive(Deserialize)]
#[serde(rename_all = "UPPERCASE")]
enum PythonLogLevelEnum {
    Trace,
    Debug,
    Info,
    Success,
    Warning,
    Error,
    Critical,
}

#[derive(Deserialize)]
struct PythonLogLevel {
    name: PythonLogLevelEnum,
}

#[derive(Deserialize)]
struct PythonLogRecord {
    level: PythonLogLevel,
}

#[derive(Deserialize)]
struct PythonLogMessage {
    text: String,
    record: PythonLogRecord,
}

impl PythonLogMessage {
    fn trace(&self) {
        match self.record.level.name {
            PythonLogLevelEnum::Trace => tracing::trace!("{}", self.text),
            PythonLogLevelEnum::Debug => tracing::debug!("{}", self.text),
            PythonLogLevelEnum::Info => tracing::info!("{}", self.text),
            PythonLogLevelEnum::Success => tracing::info!("{}", self.text),
            PythonLogLevelEnum::Warning => tracing::warn!("{}", self.text),
            PythonLogLevelEnum::Error => tracing::error!("{}", self.text),
            PythonLogLevelEnum::Critical => tracing::error!("{}", self.text),
        }
    }
}

impl TryFrom<&String> for PythonLogMessage {
    type Error = serde_json::Error;

    fn try_from(value: &String) -> Result<Self, Self::Error> {
        serde_json::from_str::<Self>(value)
    }
}

pub(crate) fn log_lines<S: Sized + BufRead>(lines: Lines<S>) {
    for line in lines.map_while(Result::ok) {
        match PythonLogMessage::try_from(&line) {
            Ok(log) => log.trace(),
            Err(_) => tracing::debug!("{line}"),
        }
    }
}
