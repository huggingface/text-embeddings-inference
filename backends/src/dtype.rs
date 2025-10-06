use std::fmt;

#[cfg(feature = "clap")]
use clap::ValueEnum;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "clap", derive(Clone, ValueEnum))]
pub enum DType {
    // Float16 is not available on accelerate
    #[cfg(any(
        feature = "python",
        all(feature = "candle", not(feature = "accelerate"))
    ))]
    Float16,
    #[cfg(any(feature = "python", feature = "candle", feature = "ort"))]
    Float32,
    #[cfg(feature = "python")]
    Bfloat16,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // Float16 is not available on accelerate
            #[cfg(any(
                feature = "python",
                all(feature = "candle", not(feature = "accelerate"))
            ))]
            DType::Float16 => write!(f, "float16"),
            #[cfg(any(feature = "python", feature = "candle", feature = "ort"))]
            DType::Float32 => write!(f, "float32"),
            #[cfg(feature = "python")]
            DType::Bfloat16 => write!(f, "bfloat16"),
            // Catch-all for impossible configurations
            #[cfg(all(
                not(feature = "python"),
                not(feature = "candle"),
                not(feature = "ort")
            ))]
            _ => {
                let _ = f;  // Suppress unused variable warning
                unreachable!("DType has no variants in this configuration")
            }
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for DType {
    fn default() -> Self {
        // Priority 1: Python feature → Bfloat16
        #[cfg(feature = "python")]
        {
            return DType::Bfloat16;
        }

        // Priority 2: Candle with Float16 support (candle + !accelerate)
        #[cfg(all(
            not(feature = "python"),
            feature = "candle",
            not(feature = "accelerate")
        ))]
        {
            return DType::Float16;
        }

        // Priority 3: Candle with accelerate OR ORT → Float32
        #[cfg(all(
            not(feature = "python"),
            any(
                all(feature = "candle", feature = "accelerate"),
                feature = "ort"
            )
        ))]
        {
            return DType::Float32;
        }

        // Unreachable: If no features are enabled, build will fail earlier
        // This branch should never execute but satisfies the compiler
        #[cfg(all(
            not(feature = "python"),
            not(feature = "candle"),
            not(feature = "ort")
        ))]
        {
            unreachable!("No DType available - build should have failed earlier");
        }
    }
}
