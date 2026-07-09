from typing import Any

import torch
import torch.nn as nn # Infraestrutura da rede
from torch.distributions import Normal # Amostragem da distribuição normal para o Actor


class ActorCriticNetwork(nn.Module):
    """Esta classe deve APENAS:
    - Representar uma rede neural
    - Receber um estado
    - Transformar a entrada em uma política e em um valor
    """

    def __init__(self, input_dim: int, camadas_ocultas: list[int], camadas_cabecas: list[int], num_actions: int, ativacao: type[nn.Module] = nn.Tanh):

        super().__init__()
        
        self.backbone = self.build_mlp(input_dim=input_dim, camadas_ocultas=camadas_ocultas, ativacao=ativacao)
        self.actor = self.build_mlp(input_dim=camadas_ocultas[-1], camadas_ocultas=camadas_cabecas, ativacao=ativacao)
        self.critic = self.build_mlp(input_dim=camadas_ocultas[-1], camadas_ocultas=camadas_cabecas, ativacao=ativacao)

        self.log_std = nn.Parameter(torch.zeros(num_actions)) # A exponencial desse parâmetro é o sigma da distribuição. É um parâmetro treinável/que o modelo aprende.

    @staticmethod
    def build_mlp(input_dim: int, camadas_ocultas: list[int], ativacao: type[nn.Module] = nn.Tanh):
        """Constrói o backbone da rede neural dado um número de entradas e um número de camadas ocultas.

        Args:
            input_dim (int): Número de dimenssões de entrada.
            camadas_ocultas (list[int]): Lista com o número de neurônios em cada camada oculta.
            ativacao (nn.Module, optional): A função de ativação. É aplicada na última camada do backbone. Defaults to nn.Tanh.

        Returns:
            nn.Sequential: O objeto de camadas da MLP
        """

        dimensoes = [input_dim] + camadas_ocultas
        camadas = []

        for i in range(len(dimensoes) - 1):
            camadas.append(nn.Linear(dimensoes[i], dimensoes[i+1]))
            camadas.append(ativacao())

        return nn.Sequential(*camadas)
    
    def forward(self, estado: torch.Tensor):
        """Realiza a passagem do estado pela rede de duas cabeças (actor e critic).

        Args:
            estado (torch.Tensor): O estado do sistema real, do ambiente

        Returns:
            tuple: A saída de cada uam das cabeças
        """

        features = self.backbone(estado)
        mu = self.actor(features)
        value = self.critic(features)

        return mu, value

    def act(self, estado: torch.Tensor, treino: bool = False):
        """Gera a decisão. Age. Toma a decisão com base na saída da rede. Se em treino, usa o sampling da distribuição.

        Args:
            estado (torch.Tensor): O estado verificado no ambiente
            treino (bool, optional): Define se está ocorrendo treino ou teste. Defaults to False.

        Returns:
            tuple: A ação a ser tomada (no contínuo), o logaritmo da probabilidade da ação, o valor do Critic e a distribuição usada (para futuro cálculo de entropia).
        """

        ### Em tese a notação correta usa: act(self, estado: torch.Tensor, treino: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Para deixar bem documentado e escrito!!!!!

        mu, value = self.forward(estado)
        std = torch.exp(self.log_std) # Obtém o fator sigma da distribuição
        dist = Normal(mu, std)
        action = mu
        if treino: # Usa a amostragem apenas em treino
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1) # Para ações multidimencionais usar: log_prob = dist.log_prob(action).sum(dim=-1) --> Soma dos logs das probabilidades.

        return action, log_prob, value, dist




