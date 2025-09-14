mod pb;
pub(crate) mod server;

use pb::tei::v1::{
    embed_server::EmbedServer, info_server::InfoServer, predict_server::PredictServer,
    rerank_server::RerankServer, similarity_server::SimilarityServer,
    tokenize_server::TokenizeServer, *,
};
