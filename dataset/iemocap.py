import os
import re
from pathlib import Path
from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset


class IEMOCAP(Dataset):
    """
    Args:
        root: For example'.../IEMOCAP_full_release' 
    """
    
    def __init__(self, root: str): 
        self.root = Path(root)

        if not os.path.isdir(self.root):
            raise RuntimeError("Dataset not found.")

        all_clip_names = set()

        self.data = []

        self.label_mapping = {}
        self.transcript_mapping = {}
        self.audio_path_mapping = {}
        self.video_path_mapping = {}

        for session in [1, 2, 3, 4, 5]:
            session_dir = self.root / f"Session{session}"

            label_dir = session_dir / "labels"
            audio_dir = session_dir / "audio"
            video_dir = session_dir / "video"
            transcription_dir = session_dir / "transcriptions"

            transcription_paths = transcription_dir.glob("*.txt")
            for transcription_path in transcription_paths:
                 with open(transcription_path, "r") as f:
                    for line in f:
                        if not line.startswith("Ses"):
                            continue
                        parts = line.split('[')
                        subparts = parts[1].split(':')

                        clip_name = parts[0].strip()
                        all_clip_names.add(clip_name)

                        transcript = subparts[1].strip()

                        self.transcript_mapping[clip_name] = transcript
                 
            audio_paths = audio_dir.glob("*/*.wav")
            for audio_path in audio_paths:
                clip_name = str(Path(audio_path).stem)
                all_clip_names.add(clip_name)
                self.audio_path_mapping[clip_name] = audio_path


            video_paths = video_dir.glob("*/*.avi")
            for video_path in video_paths:
                clip_name = str(Path(video_path).stem)
                all_clip_names.add(clip_name)
                self.video_path_mapping[clip_name] = video_path

            label_paths = label_dir.glob("*.txt")
            for label_path in label_paths:
                with open(label_path, "r") as f:
                    for line in f:
                        if not line.startswith("["):
                            continue
                        line = re.split("[\t\n]", line)
                        clip_name = line[1]
                        all_clip_names.add(clip_name)

                        label = line[2]
                        if label == 'exc':
                            label = 'hap'
                        if label not in ['ang', 'fru', 'hap', 'neu', 'sad']:
                            continue
                        self.label_mapping[clip_name] = label

        for clip_name in all_clip_names:
            if (self.label_mapping.get(clip_name) is not None and 
                self.audio_path_mapping.get(clip_name) is not None and
                self.transcript_mapping.get(clip_name) is not None and 
                self.video_path_mapping.get(clip_name) is not None):
                    self.data.append(clip_name)

def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            str:
                Label (one of "neu", "hap", "ang", "sad", "fru")
            str:
                Transcript
            str:
                Audio file path
            str:
                Video file path
        """
        clip_name = self.data[n]

        label = self.label_mapping[clip_name]
        transcript = self.transcript_mapping[clip_name]
        audio_path = self.audio_path_mapping[clip_name]
        video_path = self.video_path_mapping[clip_name]

        return (label, transcript, audio_path, video_path)


def __len__(self):
    return len(self.data)