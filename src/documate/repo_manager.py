import os
import shutil
import zipfile
from urllib.parse import urlparse
from git import Repo, GitCommandError
from io import BytesIO

class RepoManager:
    """
    Manages the cloning and extraction of code repositories from Git URLs and ZIP archives.
    """
    def __init__(self, base_clone_path: str):
        self.base_clone_path = os.path.abspath(base_clone_path)
        os.makedirs(self.base_clone_path, exist_ok=True)
        print(f"RepoManager initialized. Repositories will be stored in: {self.base_clone_path}")

    def _get_repo_name_from_url(self, repo_url: str) -> str:
        parsed_path = urlparse(repo_url).path
        repo_name = parsed_path.lstrip('/').replace('.git', '')
        return repo_name.replace('/', '_')

    def clone_repo(self, repo_url: str, pat: str | None = None) -> str | None:
        repo_name = self._get_repo_name_from_url(repo_url)
        clone_destination = os.path.join(self.base_clone_path, repo_name)
        if os.path.exists(clone_destination):
            print(f"Found existing directory. Removing '{clone_destination}' for a fresh start.")
            shutil.rmtree(clone_destination)
        print(f"Cloning '{repo_url}' into '{clone_destination}'...")
        try:
            if pat:
                parsed_url = urlparse(repo_url)
                auth_url = f"{parsed_url.scheme}://x-token-auth:{pat}@{parsed_url.netloc}{parsed_url.path}"
                Repo.clone_from(auth_url, clone_destination)
            else:
                Repo.clone_from(repo_url, clone_destination)
            print(f"✅ Successfully cloned repository to: {clone_destination}")
            return clone_destination
        except GitCommandError as e:
            print(f"❌ Error cloning repository: {e}")
            return None

    def process_zip_file(self, file_content: BytesIO, original_filename: str) -> str | None:
        dir_name = original_filename.replace('.zip', '').replace(' ', '_')
        extract_destination = os.path.join(self.base_clone_path, dir_name)
        if os.path.exists(extract_destination):
            print(f"Found existing directory. Removing '{extract_destination}' for a fresh extraction.")
            shutil.rmtree(extract_destination)
        os.makedirs(extract_destination)
        print(f"Extracting '{original_filename}' to '{extract_destination}'...")
        try:
            with zipfile.ZipFile(file_content, 'r') as zip_ref:
                top_level_dirs = {os.path.normpath(f).split(os.sep)[0] for f in zip_ref.namelist()}
                if len(top_level_dirs) == 1:
                    temp_extract_path = os.path.join(self.base_clone_path, "_temp_extract")
                    if os.path.exists(temp_extract_path):
                        shutil.rmtree(temp_extract_path)
                    zip_ref.extractall(temp_extract_path)
                    nested_dir_name = list(top_level_dirs)[0]
                    nested_dir_path = os.path.join(temp_extract_path, nested_dir_name)
                    shutil.move(nested_dir_path, extract_destination)
                    shutil.rmtree(temp_extract_path)
                else:
                    zip_ref.extractall(extract_destination)
            print(f"✅ Successfully extracted ZIP archive to: {extract_destination}")
            return extract_destination
        except Exception as e:
            print(f"❌ An unexpected error occurred during extraction: {e}")
            if os.path.exists(extract_destination):
                shutil.rmtree(extract_destination)
            return None