import subprocess
import uuid
import os
import shutil
from pathlib import Path
import sys
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_talking_video(audio_path, image_path="host.jpg", output_path=None):
    """
    Generate a talking video from audio + image using MuseTalk inference.
    """
    if output_path is None:
        output_path = Path("talking_news.mp4")

    try:
        # Paths
        musetalk_dir = Path(os.getcwd()) / "MuseTalk"
        venv_python = Path(sys.executable).resolve()
        image_path = Path(image_path).resolve()
        audio_path = Path(audio_path).resolve()
        output_path = Path(output_path).resolve()

        # Sanitize paths to remove non-ASCII characters
        image_path_str = image_path.as_posix().encode('ascii', 'ignore').decode('ascii')
        audio_path_str = audio_path.as_posix().encode('ascii', 'ignore').decode('ascii')
        output_path_str = output_path.as_posix().encode('ascii', 'ignore').decode('ascii')
        logger.info(f"Sanitized image path: {image_path_str}")
        logger.info(f"Sanitized audio path: {audio_path_str}")
        logger.info(f"Sanitized output path: {output_path_str}")

        # Validate inputs
        if not musetalk_dir.exists():
            raise FileNotFoundError("MuseTalk directory not found. Clone it first.")
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        if not (musetalk_dir / "configs/inference/test.yaml").exists():
            raise FileNotFoundError("MuseTalk config test.yaml not found.")
        if not (musetalk_dir / "models/musetalkV15/unet.pth").exists():
            raise FileNotFoundError("MuseTalk model unet.pth not found.")

        # Check omegaconf
        try:
            import omegaconf
            logger.info(f"omegaconf version: {omegaconf.__version__}")
        except ImportError:
            raise ImportError("omegaconf not installed. Run: pip install omegaconf")

        # Create temporary config file
        temp_config = musetalk_dir / "configs" / "inference" / f"temp_config_{uuid.uuid4().hex}.yaml"
        temp_config.parent.mkdir(parents=True, exist_ok=True)
        temp_output = musetalk_dir / "results" / "tmp" / f"output_{uuid.uuid4().hex}.mp4"
        temp_output.parent.mkdir(parents=True, exist_ok=True)

        config_data = {
            "task_1": {
                "video_path": image_path_str,
                "audio_path": audio_path_str,
                "result_name": temp_output.as_posix()
            }
        }
        with open(temp_config, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, allow_unicode=False)
        logger.info(f"Created temp config: {temp_config}")

        # Command: Use supported arguments for MuseTalk
        cmd = [
            str(venv_python), "-m", "scripts.inference",
            "--inference_config", str(temp_config),
            "--result_dir", str(musetalk_dir / "results/tmp"),
            "--unet_model_path", str(musetalk_dir / "models/musetalkV15/unet.pth"),
            "--unet_config", str(musetalk_dir / "models/musetalkV15/musetalk.json"),
            "--version", "v15",
            "--ffmpeg_path", "ffmpeg",
            "--batch_size", "8",
            "--fps", "25"
        ]

        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            cwd=musetalk_dir,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )

        logger.info(f"MuseTalk output: {result.stdout}")
        if result.stderr:
            logger.warning(f"MuseTalk warnings: {result.stderr}")

        # Copy temp output to project root
        if temp_output.exists():
            shutil.copy2(temp_output, output_path)
            logger.info(f"✅ Talking video saved as {output_path}")
            return str(output_path)
        else:
            logger.error("❌ No output video generated.")
            return None

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ MuseTalk subprocess failed: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"❌ MuseTalk generation failed: {e}")
        return None
    finally:
        # Clean up temporary config
        if 'temp_config' in locals() and temp_config.exists():
            try:
                os.remove(temp_config)
                logger.info(f"Cleaned up temporary config: {temp_config}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary config: {e}")