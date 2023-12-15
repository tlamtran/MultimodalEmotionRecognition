import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from transformers import Wav2Vec2Model, RobertaModel, VivitModel

class Modality(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        self.weights = nn.Parameter(torch.randn(12))
        self.feed_forward = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )
        self.classifier = nn.Linear(768, 5)


    def forward(self, inputs):
        model_outputs = self.model(**inputs)
        hidden_states = model_outputs.hidden_states[1:]
        hidden_states = self.weighted_average(hidden_states, self.weights)
        x = self.feed_forward(hidden_states)
        x = self.classifier(x)
        return x

    @staticmethod
    def weighted_average(hidden_states, weights):
        hidden_states = torch.stack(hidden_states, axis=0)[:, :, 0]
        weights = F.softmax(weights, dim=0)
        weights = weights.view(12, 1, 1)
        weighted_hidden_states = hidden_states * weights
        return torch.sum(weighted_hidden_states, dim=0)


class MultimodalClassificationHead(nn.Module):
    def __init__(self, audio_modality=False, text_modality=False, video_modality=False):
        super().__init__()
        self.modalities = nn.ModuleDict()

        if audio_modality:
            self.modalities['audio'] = Modality(Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True))

        if text_modality:
            self.modalities['text'] = Modality(RobertaModel.from_pretrained("roberta-base", output_hidden_states=True))

        if video_modality:        
            self.modalities['video'] = Modality(VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", output_hidden_states=True))


    def forward(self, audio_inputs=None, text_inputs=None, video_inputs=None):
        outputs = []

        if 'audio' in self.modalities and audio_inputs is not None:
            outputs.append(self.modalities['audio'](audio_inputs))

        if 'text' in self.modalities and text_inputs is not None:
            outputs.append(self.modalities['text'](text_inputs))

        if 'video' in self.modalities and video_inputs is not None:
            outputs.append(self.modalities['video'](video_inputs))

        if not outputs:
            raise ValueError("No valid modalities provided")

        x = torch.stack(outputs)
        x = torch.mean(x, dim=0)

        return x