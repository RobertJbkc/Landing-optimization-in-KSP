### Este arquivo responda a pergunta: Como atualizar a Actor-Critic usando as experiências armazenadas no Rollout Buffer?
# Conhece os seguintes módulos:
# - ActorCriticNetwork
# - RolloutBuffer
# - Optimizer

import torch
import torch.nn as nn
from torch.distributions import Normal # Amostragem da distribuição normal para o Actor
from rocket_lander.models.actor_critic import ActorCriticNetwork
from rocket_lander.memory.rollout_buffer import RolloutBuffer
import numpy as np



class PPO():
    """
    Deve conhecer, APENAS:
    - ActorCriticNetwork
    - RolloutBuffer
    - Optimizer
    """

    def __init__(self, rede: ActorCriticNetwork, lr: float = 5e-4, gamma: float = 0.99, lambda_gae: float = 0.95, epsilon_clip: float = 0.2, coef_entropia: float = 0.5, coef_valor: float = 0.5, epocas: int = 20, batch_size: int = 64):
        self.rede = rede
        self.lr = lr
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.epsilon_clip = epsilon_clip
        self.coef_entropia = coef_entropia
        self.coef_valor = coef_valor
        self.epocas = epocas
        self.batch_size = batch_size

        # Otimizador
        self.optimizer = torch.optim.Adam(
            self.rede.parameters(),
            lr=self.lr
        )

        self.perda = nn.MSELoss()

    
    def calcula_vantagens(self, recompensas: torch.Tensor, valores: torch.Tensor, dones: torch.Tensor, proximo_valor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calcula a vantagem e o retorno dado uma ação. "mask" inverte o sentido de "done", onde 1 representa fim do episódio após ação, portanto sem pr'
        oximo valor.

        Args:
            recompensas (torch.Tensor): A recompença calculada ou aproximada por algo.
            valores (torch.Tensor): O valor da bonicidade da ação tomado, vem do Critic.
            dones (torch.Tensor): Resposta da pergunta: "O episódio terminou (1) ou não (0) após a tomada da ação".
            proximo_valor (torch.Tensor): É o valor que entra no caso do cálculo do delta do último ponto, pois não há um proximo.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A vantagem de ter executado uma ação e o retorno (vantagem + valores)
        """

        vantagens = torch.zeros_like(valores)
        gae = torch.zeros_like(proximo_valor)

        for t in reversed(range(len(recompensas))):
            if t == len(recompensas) - 1:
                proximo_valor = proximo_valor
            
            else:
                proximo_valor = valores[t+1]

            dones_f = dones.float() # Converter para float
            mask = 1 - dones_f[t]
            delta = recompensas[t] + (self.gamma * mask * proximo_valor) - valores[t]
            gae = delta + (self.gamma * self.lambda_gae * mask * gae)
            vantagens[t] = gae
        
        retornos = vantagens + valores

        return vantagens, retornos
    
    def atualizar(self, buffer: RolloutBuffer, proximo_valor: torch.Tensor):
        """Responde à pergunta: Como usar o rollout armazenado para melhorar a política?

        Args:
            buffer (RolloutBuffer): O buffer da dados gerado pela classe RolloutBuffer.
            proximo_valor (torch.Tensor): O valor que se aplica para ser o seguinte em relação ao último valor do buffer.
        """

        # converter para tensores
        dados = buffer.to_tensors()
        # Calcula vantagens e retornos
        vantagens, retornos = self.calcula_vantagens(recompensas=dados['recompensas'], valores=dados['valores'], dones=dados['dones'], proximo_valor=proximo_valor)
        vantagens = vantagens.detach()
        retornos = retornos.detach()
        # # Normaliza as vantagens
        # vantagens = (vantagens - vantagens.mean()) / (vantagens.std() + 1e-8) # Evitar divisão por zero

        for _ in range(self.epocas):
            indices = torch.randperm(buffer.size)
            for inicio in range(0, buffer.size, self.batch_size):
                fim = inicio + self.batch_size
                # Divide em mini-batches
                batch = indices[inicio:fim]

                # Extrair as informações do batch, com base na posição do buffer (índice)
                estados_batch = dados['estados'][batch]
                acoes_batch = dados['acoes'][batch]
                log_probs_batch = dados['log_probs'][batch]
                vantagens_batch = vantagens[batch]
                retornos_batch = retornos[batch]
            
                # Fazer o forward na rede (está implicito, interno na calc_loss)
                # Calcular as perdas
                loss, _, _, _ = self.calc_loss(estados=estados_batch, acoes=acoes_batch, log_probs_antigos=log_probs_batch, vantagens=vantagens_batch, retornos=retornos_batch)

                # Atualizar os parâmetros da rede
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Limpar o buffer ao final de tudo
        print('[AAAAA] Atualizou!!!!! #########################')
        buffer.clear()

    def calc_loss(self, estados: torch.Tensor, acoes: torch.Tensor, log_probs_antigos: torch.Tensor, vantagens: torch.Tensor, retornos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calcula a função de perda do PPO, composta de três termos. "actor_loss" otimiza a política. "critic_loss" otimiza a estimativa de valor. "entropy" incentiva a exploração.

        Args:
            estados (torch.Tensor): Estados observados pela agente.
            acoes (torch.Tensor): Ações tomadas durante o rollout.
            log_probs_antigos (torch.Tensor): Logaritmo da probabilidade das ações segundo a política antiga.
            vantagens (torch.Tensor): Estimativa da vantagem calculada pelo Generalized Advantage Estimation (GAE).
            retornos (torch.Tensor): Valor alvo para o treinamento do Crictic.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tupla com a perda total e a perda de cada parte do algoritmo, Actor, Critic e entropia.
        """
        
        mu, valores = self.rede(estados) # Novo valor do Critic e nova distribuição de ações. Internamente o torch chama a "forward()"
        dist = Normal(mu, torch.exp(self.rede.log_std))
        # Qual a probabilidadede das ações antigas segundo a nova política?
        novo_log_prob = dist.log_prob(acoes).sum(dim=-1)
        razao = torch.exp(novo_log_prob - log_probs_antigos) # Comparação on novo com o antigo log_prob

        ##### Loss do Actor
        razao_clipada = torch.clamp(razao, 1-self.epsilon_clip, 1+self.epsilon_clip)
        actor_loss = torch.min(razao * vantagens, razao_clipada * vantagens).mean()

        ##### Loss do Critic
        critic_loss = self.coef_valor * self.perda(valores, retornos)

        ##### Entropia - Favorecer a exploração (deve ser maximizada)
        entropia_loss = - self.coef_entropia * dist.entropy().sum(dim=-1).mean()

        ##### Combinar
        loss = actor_loss + critic_loss + entropia_loss
        print(f'Loss {loss}, Actor {actor_loss}, Critic {critic_loss}, Entrpy {entropia_loss}')

        return loss, actor_loss, critic_loss, entropia_loss