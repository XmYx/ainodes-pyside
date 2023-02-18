# this scripts installs necessary requirements and launches main program in webui.py
import subprocess
import os
import sys
import importlib.util
import shlex
import platform
import ctypes
import colorama
from termcolor import colored


dir_repos = "src"
python = sys.executable
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")


def extract_arg(args, name):
    return [x for x in args if x != name], name in args


def run(command, desc=None, errdesc=None):
    if desc is not None:
        print('desc', desc)

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def check_run(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return result.returncode == 0


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def repo_dir(name):
    return os.path.join(dir_repos, name)


def run_python(code, desc=None, errdesc=None):
    return run(f'"{python}" -c "{code}"', desc, errdesc)


def run_pip(args, desc=None):
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {args} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")


def check_run_python(code):
    return check_run(f'"{python}" -c "{code}"')


def git_clone(url, dir, name, commithash=None):
    # TODO clone into temporary dir and move if successful

    if os.path.exists(dir):
        #sys.path.append(os.path.join(os.getcwd(),dir))
        if commithash is None:
            return

        current_hash = run(f'"{git}" -C {dir} rev-parse HEAD', None, f"Couldn't determine {name}'s hash: {commithash}").strip()
        if current_hash == commithash:
            return

        run(f'"{git}" -C {dir} fetch', f"Fetching updates for {name}...", f"Couldn't fetch {name}")
        run(f'"{git}" -C {dir} checkout {commithash}', f"Checking out commit for {name} with hash: {commithash}...", f"Couldn't checkout commit {commithash} for {name}")
        return

    run(f'"{git}" clone "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}")

    if commithash is not None:
        run(f'"{git}" -C {dir} checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")

        
def version_check(commit):
    try:
        import requests
        commits = requests.get('https://github.com/osi1880vr/ainodes-pyside-dev/branches/dev').json()
        if commit != "<none>" and commits['commit']['sha'] != commit:
            print("--------------------------------------------------------")
            print("| You are not up to date with the most recent release. |")
            print("| Consider running `git pull` to update.               |")
            print("--------------------------------------------------------")
        elif commits['commit']['sha'] == commit:
            print("You are up to date with the most recent release.")
        else:
            print("Not a git clone, can't perform version check.")
    except Exception as e:
        print("versiom check failed",e)

        
def prepare_enviroment():
    torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")
    commandline_args = os.environ.get('COMMANDLINE_ARGS', "")

    gfpgan_package = os.environ.get('GFPGAN_PACKAGE', "git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379")
    clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
    openclip_package = os.environ.get('OPENCLIP_PACKAGE',
                                      "git+https://github.com/mlfoundations/open_clip.git@bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b")

    deepdanbooru_package = os.environ.get('DEEPDANBOORU_PACKAGE', "git+https://github.com/KichangKim/DeepDanbooru.git@d91a2963bf87c6a770d74894667e9ffa9f6de7ff")

    xformers_windows_package = os.environ.get('XFORMERS_WINDOWS_PACKAGE', 'https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/torch13/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl')

    stable_diffusion_repo = os.environ.get('STABLE_DIFFUSION_REPO', "https://github.com/CompVis/stable-diffusion.git")
    taming_transformers_repo = os.environ.get('TAMING_REANSFORMERS_REPO', "https://github.com/CompVis/taming-transformers.git")
    k_diffusion_repo = os.environ.get('K_DIFFUSION_REPO', 'https://github.com/crowsonkb/k-diffusion.git')
    codeformer_repo = os.environ.get('CODEFORMET_REPO', 'https://github.com/sczhou/CodeFormer.git')
    blip_repo = os.environ.get('BLIP_REPO', 'https://github.com/salesforce/BLIP.git')

    real_esrgan_repo = os.environ.get('REAL_ESRGAN_REPO', "https://github.com/xinntao/Real-ESRGAN.git")
    adabins_repo = os.environ.get('ADABINS_REPO', "https://github.com/osi1880vr/AdaBins.git")
    midas_repo = os.environ.get('MIDAS_REPO', 'https://github.com/isl-org/MiDaS.git')
    pytorch_lite_repo = os.environ.get('PYTORCH_LITE_REPO', 'https://github.com/osi1880vr/pytorch3d-lite.git')
    impro_aesthetic_repo = os.environ.get('IMPRO_AESTHETIC_REPO', 'https://github.com/christophschuhmann/improved-aesthetic-predictor.git')
    volta_ml_repo = os.environ.get('VOLTA_ML_REPO', 'https://github.com/VoltaML/voltaML-fast-stable-diffusion.git')

    stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc")
    taming_transformers_commit_hash = os.environ.get('TAMING_TRANSFORMERS_COMMIT_HASH', "24268930bf1dce879235a7fddd0b2355b84d7ea6")
    k_diffusion_commit_hash = os.environ.get('K_DIFFUSION_COMMIT_HASH', "5b3af030dd83e0297272d861c19477735d0317ec")
    codeformer_commit_hash = os.environ.get('CODEFORMER_COMMIT_HASH', "c5b4593074ba6214284d6acd5f1719b6c5d739af")
    blip_commit_hash = os.environ.get('BLIP_COMMIT_HASH', "48211a1594f1321b00f14c9f7a5b4813144b2fb9")
    volta_ml_hash = os.environ.get('VOLTA_ML__HASH', "303d5f8df54f58987818722226a6398a9aed8aa6")

    real_esrgan_commit_hash = os.environ.get('REAL_ESRGAN_COMMIT_HASH', "64ad194ddaf9c4d8c4b0d1b98cac6d89d3ea0d11")
    adabins_commit_hash = os.environ.get('ADABINS_COMMIT_HASH', "4524615236f5f486381fac2f9c624f20dedf324f")
    midas_commit_hash = os.environ.get('MIDAS_COMMIT_HASH', "66882994a432727317267145dc3c2e47ec78c38a")
    pytorch_litet_hash = os.environ.get('PYTORCH_LITE_COMMIT_HASH', "4070975c1d6e4de7c87848a53b603f6b29711e55")
    impro_aesthetic_hash = os.environ.get('IMPRO_AESTHETIC_COMMIT_HASH', "fe88a163f4661b4ddabba0751ff645e2e620746e")

    sys.argv += shlex.split(commandline_args)
    test_argv = [x for x in sys.argv if x != '--tests']

    sys.argv, skip_torch_cuda_test = extract_arg(sys.argv, '--skip-torch-cuda-test')
    sys.argv, reinstall_xformers = extract_arg(sys.argv, '--reinstall-xformers')
    sys.argv, update_check = extract_arg(sys.argv, '--update-check')
    sys.argv, run_tests = extract_arg(sys.argv, '--tests')
    xformers = True #'--xformers' in sys.argv
    deepdanbooru = '--deepdanbooru' in sys.argv
    ngrok = '--ngrok' in sys.argv

    try:
        commit = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        commit = "<none>"

    print(f"Python {sys.version}")
    print(f"Commit hash: {commit}")

    #if not is_installed("torch") or not is_installed("torchvision") or not torch.cuda.is_available():
    try:
        cudart = ctypes.CDLL('cudart')
        cudart.cudaGetDeviceCount.restype = int
        cudart.cudaGetDeviceCount()
    except:
        print('your version of torch is not cuda enabled, therefore we now enforce a cuda enabled version of torch')
        run(f'"{python}" -m {torch_command}', f"Installing torch and torchvision", "Couldn't install torch")


    if not skip_torch_cuda_test:
        run_python("import torch; assert torch.cuda.is_available(), 'Torch is not able to use GPU; add --skip-torch-cuda-test to COMMANDLINE_ARGS variable to disable this check'")

    if not is_installed("gfpgan"):
        run_pip(f"install {gfpgan_package}", "gfpgan")

    if not is_installed("clip"):
        run_pip(f"install {clip_package}", "clip")

    if not is_installed("open_clip"):
        run_pip(f"install {openclip_package}", "open_clip")

    if (not is_installed("xformers") or reinstall_xformers) and xformers:
        if platform.system() == "Windows":
            if platform.python_version().startswith("3.10"):
                run_pip(f"install -U -I --no-deps {xformers_windows_package}", "xformers")
            else:
                print("Installation of xformers is not supported in this version of Python.")
                #if not is_installed("xformers"):
                #    exit(0)
        elif platform.system() == "Linux":
            run_pip("install xformers", "xformers")

    if not is_installed("deepdanbooru") and deepdanbooru:
        run_pip(f"install {deepdanbooru_package}#egg=deepdanbooru[tensorflow] tensorflow==2.10.0 tensorflow-io==0.27.0", "deepdanbooru")

    if not is_installed("pyngrok") and ngrok:
        run_pip("install pyngrok", "ngrok")

    os.makedirs(dir_repos, exist_ok=True)

    # we don't fetch this as we maintain our own LDM
    # git_clone(stable_diffusion_repo, repo_dir('stable-diffusion'), "Stable Diffusion", stable_diffusion_commit_hash)
    git_clone(taming_transformers_repo, repo_dir('taming-transformers'), "Taming Transformers", taming_transformers_commit_hash)
    git_clone(k_diffusion_repo, repo_dir('k-diffusion'), "K-diffusion", k_diffusion_commit_hash)
    git_clone(codeformer_repo, repo_dir('CodeFormer'), "CodeFormer", codeformer_commit_hash)
    git_clone(blip_repo, repo_dir('BLIP'), "BLIP", blip_commit_hash)

    git_clone(real_esrgan_repo, repo_dir('realesrgan'), "Real ESRGAN", real_esrgan_commit_hash)
    git_clone(adabins_repo, repo_dir('AdaBins'), "AdaBins", adabins_commit_hash)
    git_clone(midas_repo, repo_dir('MiDaS'), "MiDaS", midas_commit_hash)
    git_clone(pytorch_lite_repo, repo_dir('pytorch3d-lite'), "pytorch3d-lite", pytorch_litet_hash)
    git_clone(impro_aesthetic_repo, repo_dir('improved-aesthetic-predictor'), "improved-aesthetic-predictor", impro_aesthetic_hash)
    #git_clone(volta_ml_repo, repo_dir('volta-ml'), "volta-ml", volta_ml_hash)

    if not is_installed("lpips"):
        run_pip(f"install -r {os.path.join(repo_dir('CodeFormer'), 'requirements.txt')}", "requirements for CodeFormer")

    print(f'running: install -r {requirements_file} requirements for SD UI' )
    run_pip(f"install -r {requirements_file}", "requirements for SD UI")


    #run_pip(f"install git+https://github.com/smirkingface/stable-diffusion", "smirkingface")

    #if update_check:
    #    version_check(commit)
    
    if "--exit" in sys.argv:
        print("Exiting because of --exit argument")
        exit(0)


def start_sdui():
    print(f"Launching SD UI")
    import frontend.startup
    print(colored("Installation part is done now we run the application, please stay patient.", "green"))
    print(colored("You might see a few warnings about No instance of QPyDesignerCustomWidgetCollection was found.", "green"))
    print(colored("Just ignore those.", "green"))
    frontend.startup.run_app()

if __name__ == "__main__":
    colorama.init()
    print(colored("The main Packages will get installed now, please be patient.\n", "red"))
    prepare_enviroment()
    import backend.paths
    start_sdui()
