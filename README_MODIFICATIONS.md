# Text Embeddings Inference - SM87 适配版本

## 修改内容

本版本基于 text-embeddings-inference 项目，针对 NVIDIA Jetson Orin (SM87) 和 L4 GPU (SM89) 进行了适配，并集成了以下社区 PR：

### 1. SM87/SM89 CUDA 支持
- 支持 NVIDIA Jetson Orin AGX (compute capability 8.7)
- 支持 NVIDIA L4 GPU (compute capability 8.9)
- 修改文件：
  - `Dockerfile-cuda-all`
  - `cuda-all-entrypoint.sh`
  - `backends/candle/src/compute_cap.rs`

### 2. PR #730: Qwen3 Reranker 支持
- 添加 Qwen3 分类头用于重排序任务
- 实现模板格式化系统支持聊天格式
- 修改文件：
  - `backends/candle/src/models/qwen3.rs`
  - `core/src/templates.rs` (新增)
  - `core/src/lib.rs`

### 3. PR #787: 批处理通知性能优化
- 使用 AtomicUsize 计数器优化批处理场景的线程通知
- 仅在批处理最后一个请求时触发通知，减少不必要的 notify_one() 调用
- 修改文件：
  - `core/src/infer.rs`
  - `router/src/http/server.rs`
  - `router/src/grpc/server.rs`

### 4. PR #753: GeLU 激活函数一致性修复
- 将 Gelu 从近似版本 (gelu) 改为精确版本 (gelu_erf)
- 添加 NewGelu 变体保持向后兼容
- 修改文件：
  - `backends/candle/src/layers/linear.rs`

### 5. PR #790: StaticEmbedding 模型支持
- 支持 sentence-transformers 的 0_StaticEmbedding/ 目录结构
- 添加模型权重和 tokenizer 的 fallback 加载逻辑
- 为 StaticEmbedding 模型默认使用 Mean pooling
- 修改文件：
  - `backends/candle/src/models/static_embedding.rs` (新增)
  - `backends/candle/src/lib.rs`
  - `backends/src/lib.rs`
  - `core/src/download.rs`
  - `router/src/lib.rs`

### 6. PR #746: DebertaV2 序列分类支持
- 添加完整的 DebertaV2 模型实现
- 支持序列分类任务（如 Llama Prompt Guard）
- 支持 CPU 和 CUDA 设备
- 修改文件：
  - `backends/candle/src/models/debertav2.rs` (新增)
  - `backends/candle/src/lib.rs`
  - `backends/candle/src/models/mod.rs`

## 编译验证

所有修改已通过编译检查：
```bash
cargo check --all-targets
Finished `dev` profile [unoptimized + debuginfo] target(s) in 23.76s
```

## 部署说明

### 构建 Docker 镜像（支持 SM87/SM89）
```bash
docker build -f Dockerfile-cuda-all -t tei-sm87:latest .
```

### 运行示例
```bash
docker run --gpus all -p 8080:80 \
  -v $PWD/data:/data \
  tei-sm87:latest \
  --model-id BAAI/bge-large-zh-v1.5 \
  --pooling mean
```

## 修改日期
2026年1月5日
