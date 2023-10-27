use crate::layers::HiddenAct;
use candle::{Device, Result, Tensor};
use lazy_static::lazy_static;

#[cfg(feature = "cuda")]
use candle_cublaslt::{fused_batch_matmul, fused_matmul, Activation, CublasLt};

lazy_static! {
    pub static ref CUBLASLT: Option<CublasLtWrapper> = {
        match Device::cuda_if_available(0) {
            Ok(device) => {
                #[cfg(feature = "cuda")]
                {
                    Some(CublasLtWrapper {
                        cublaslt: CublasLt::new(&device).unwrap(),
                    })
                }
                #[cfg(not(feature = "cuda"))]
                {
                    None
                }
            }
            Err(_) => None,
        }
    };
}

#[derive(Debug, Clone)]
pub struct CublasLtWrapper {
    #[cfg(feature = "cuda")]
    pub cublaslt: CublasLt,
}

impl CublasLtWrapper {
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
        out: Option<&Tensor>,
        alpha: Option<f32>,
        beta: Option<f32>,
        bias: Option<&Tensor>,
        act: Option<HiddenAct>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let act = act.clone().map(|a| match a {
                HiddenAct::Gelu => Activation::Gelu,
                HiddenAct::Relu => Activation::Relu,
            });

            fused_matmul(
                &a,
                &b,
                out,
                alpha,
                beta,
                bias,
                act.clone(),
                self.cublaslt.clone(),
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle::bail!("`cuda` feature is not enabled")
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
        out: Option<&Tensor>,
        alpha: Option<f32>,
        beta: Option<f32>,
        bias: Option<&Tensor>,
        act: Option<HiddenAct>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let act = act.clone().map(|a| match a {
                HiddenAct::Gelu => Activation::Gelu,
                HiddenAct::Relu => Activation::Relu,
            });

            fused_batch_matmul(
                &a,
                &b,
                out,
                alpha,
                beta,
                bias,
                act.clone(),
                self.cublaslt.clone(),
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle::bail!("`cuda` feature is not enabled")
        }
    }
}
