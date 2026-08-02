from huggingface_hub import HfApi
import os

# Constants
HF_SPACE_REPO = "bpinto16/Predictive-Maintenance-HFSpace"

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="predictive_maintenance_project/deployment",     # the local folder containing your files
    repo_id=HF_SPACE_REPO,          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
