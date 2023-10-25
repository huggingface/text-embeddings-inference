use candle::{Device, Result, Tensor, D};
use lazy_static::lazy_static;
use serde::Deserialize;

#[cfg(feature = "cuda")]
use candle_cublaslt::{fused_matmul, Activation, CublasLt};

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
    pub fn matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
        bias: Option<&Tensor>,
        act: Option<HiddenAct>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let act = act.clone().map(|a| match a {
                HiddenAct::Gelu => Activation::Gelu,
                HiddenAct::Relu => Activation::Relu,
            });

            fused_matmul(&a, &b, bias, act.clone(), self.cublaslt.clone())
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle::bail!("`cuda` feature is not enabled")
        }
    }
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    Relu,
}

#[derive(Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    act: Option<HiddenAct>,
    span: tracing::Span,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>, act: Option<HiddenAct>) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "linear");

        Self {
            weight,
            bias,
            act,
            span,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        #[allow(unused)]
        if let (Device::Cuda(_), Some(cublaslt)) = (x.device(), &*CUBLASLT) {
            // fused matmul requires x to be dims2
            let mut final_shape = x.dims().to_vec();
            final_shape.pop();
            final_shape.push(self.weight.dims()[0]);

            let x = x.flatten_to(D::Minus2)?;
            let result = cublaslt.matmul(&self.weight, &x, self.bias.as_ref(), self.act.clone())?;
            result.reshape(final_shape)
        } else {
            let w = match x.dims() {
                &[bsize, _, _] => self.weight.broadcast_left(bsize)?.t()?,
                _ => self.weight.t()?,
            };
            let x = x.matmul(&w)?;
            let x = match &self.bias {
                None => Ok(x),
                Some(bias) => x.broadcast_add(bias),
            }?;
            if let Some(act) = &self.act {
                match act {
                    HiddenAct::Gelu => x.gelu(),
                    HiddenAct::Relu => x.relu(),
                }
            } else {
                Ok(x)
            }
        }
    }
}
