use std::{fmt, str::FromStr};

#[cfg(feature = "clap")]
use clap::ValueEnum;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "clap", derive(Clone, ValueEnum))]
pub enum DType {
    #[cfg(any(
        feature = "python",
        all(feature = "candle", not(feature = "accelerate"))
    ))]
    Float16,
    #[cfg(any(feature = "python", feature = "candle", feature = "ort"))]
    Float32,
    #[cfg(any(
        feature = "python",
        all(feature = "candle", any(feature = "metal", feature = "cuda"))
    ))]
    Bfloat16,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DTypeParseError;

impl FromStr for DType {
    type Err = DTypeParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let dtype = match s {
            "float32" => DType::Float32,
            #[cfg(any(
                feature = "python",
                all(feature = "candle", not(feature = "accelerate"))
            ))]
            "float16" => DType::Float16,
            #[cfg(any(
                feature = "python",
                all(feature = "candle", any(feature = "metal", feature = "cuda"))
            ))]
            "bfloat16" => DType::Bfloat16,
            _ => return Err(DTypeParseError),
        };

        Ok(dtype)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            #[cfg(any(
                feature = "python",
                all(feature = "candle", not(feature = "accelerate"))
            ))]
            DType::Float16 => write!(f, "float16"),
            #[cfg(any(feature = "python", feature = "candle", feature = "ort"))]
            DType::Float32 => write!(f, "float32"),
            #[cfg(any(
                feature = "python",
                all(feature = "candle", any(feature = "metal", feature = "cuda"))
            ))]
            DType::Bfloat16 => write!(f, "bfloat16"),
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for DType {
    fn default() -> Self {
        #[cfg(any(feature = "accelerate", feature = "mkl", feature = "ort"))]
        {
            DType::Float32
        }
        #[cfg(not(any(
            feature = "accelerate",
            feature = "mkl",
            feature = "ort",
            feature = "python"
        )))]
        {
            DType::Float16
        }
    }
}
