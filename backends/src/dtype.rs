use std::fmt;

#[cfg(feature = "clap")]
use clap::ValueEnum;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "clap", derive(Clone, ValueEnum))]
pub enum DType {
    #[cfg(any(feature = "python", feature = "candle"))]
    Float16,
    #[cfg(any(feature = "python", feature = "candle"))]
    Float32,
    // #[cfg(feature = "candle")]
    // Q6K,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            #[cfg(any(feature = "python", feature = "candle"))]
            DType::Float16 => write!(f, "float16"),
            #[cfg(any(feature = "python", feature = "candle"))]
            DType::Float32 => write!(f, "float32"),
            // #[cfg(feature = "candle")]
            // DType::Q6K => write!(f, "q6k"),
        }
    }
}
