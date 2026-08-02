"""
Implemeta a estruturna necessária para criar uma rede neural do tipo MLP. O fim da implementação é construir uma rede neural de duas cabeças.

Na rede de duas cabeças, o backbone tende a ser mais denso para realizar a grande maior parte do processamento de informação (estados).
"""

from typing import Any
import torch
import torch.nn as nn
from torch.distributions import Normal # Amostragem da distribuição normal para o Actor


class ActorCriticNetwork(nn.Module):

    def __init__(self, input_dim: int, camadas_ocultas: list[int], output_dim: int, camadas_cabecas: list[int], num_actions: int, ativacao: type[nn.Module] = nn.Tanh, ativacao_saida: type[nn.Module] = nn.Sigmoid):

        super().__init__()
        
        self.backbone = self.build_mlp(input_dim=input_dim, camadas_ocultas=camadas_ocultas, output_dim=output_dim, ativacao=ativacao)

        self.actor = self.build_mlp(input_dim=output_dim, camadas_ocultas=camadas_cabecas, output_dim=num_actions, ativacao=nn.Tanh, ativacao_saida=nn.Tanh)
        self.critic = self.build_mlp(input_dim=output_dim, camadas_ocultas=camadas_cabecas, output_dim=num_actions, ativacao=ativacao, ativacao_saida=None)

        self.log_std = nn.Parameter(torch.tensor([-3.0])) # A exponencial desse parâmetro é o sigma da distribuição. É um parâmetro treinável/que o modelo aprende.


    @staticmethod
    def build_mlp(input_dim: int, camadas_ocultas: list[int], output_dim: int, ativacao: type[nn.Module] = nn.Tanh, ativacao_saida: type[nn.Module] | None = None) -> nn.Sequential:
        """Monta a arquitetura de uma rede neural tipo MLP dado um número de entradas, um conjunto de camadas ocultas, um número de saídas e uma função de ativação.
        A mais, uma função de ativação pode ser defininda na saída de rede.

        Args:
            input_dim (int): O número de entradas da rede.
            camadas_ocultas (list[int]): Uma lista representando, em cada índice, o número de neurônios na camada da camada oculta.
            output_dim (int): O número de saídas da rede.
            ativacao (type[nn.Module], optional): A função de ativação da rede. Defaults to nn.Tanh.
            ativacao_saida (type[nn.Module] | None, optional): Uma função que pode ser aplicada na saída. Defaults to None.

        Returns:
            nn.Sequential: A rede neural.
        """

        arquitetura = []

        # ===== Priemira camada
        arquitetura.append(nn.Linear(input_dim, camadas_ocultas[0]))
        arquitetura.append(ativacao())

        # ===== Camadas ocultas
        for i in range(1, len(camadas_ocultas)):
            arquitetura.append(nn.Linear(camadas_ocultas[i - 1], camadas_ocultas[i]))
            arquitetura.append(ativacao())

        # ===== Camada de saída
        arquitetura.append(nn.Linear(camadas_ocultas[-1], output_dim))
        if ativacao_saida != None:
            arquitetura.append(ativacao_saida())

        return nn.Sequential(*arquitetura)
    
    def forward(self, estado: torch.Tensor):
        """Realiza a passagem do estado pela rede de duas cabeças (Cctor e Critic).

        Args:
            estado (torch.Tensor): O estado do sistema real, do ambiente.

        Returns:
            tuple: A saída de cada ums das cabeças
        """

        features = self.backbone(estado)

        mu = self.actor(features)
        value = self.critic(features)

        return mu, value

    def act(self, estado: torch.Tensor, treino: bool = False):
        """Gera a decisão. Age. Toma a decisão com base na saída da rede. Se em treino, usa o sampling da distribuição, que, se calibrado, leva a uma boa exploração.

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
        # print('Loging de ações tomadas')
        # print(mu.mean(), mu.std())
        # print(torch.exp(self.log_std).item())
        # print(action.mean(), action.std())
        print(f'Mu: {mu[0]:.3f}, Value: {value[0]:.3f}, log_std: {self.log_std.item():.3f} std: {std[0]:.3f}')
        if treino: # Usa a amostragem apenas em treino
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1) # Para ações multidimencionais usar: log_prob = dist.log_prob(action).sum(dim=-1) --> Soma dos logs das probabilidades.

        return action, log_prob, value, dist




