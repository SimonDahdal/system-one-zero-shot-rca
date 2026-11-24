from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="phamquiluan/RCAEval",
    repo_type="dataset",
    allow_patterns=["re2*/**", "cases.parquet"],
    local_dir="data"
)
