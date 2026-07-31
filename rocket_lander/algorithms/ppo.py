"""
Este scrip implementa a lógica que informa como atualizar a rede Actor-Critic baseado nas experiências armazenadas no Rollout Buffer.

Conhece os módulos:
- ActorCriticNetwork: Que implementa a rede neural.
- RolloutBuffer: Que armazena as experiências.
- Optimizaer: Que otimiza a rede via métodos de gradiente.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal # Amostragem da distribuição normal para o Actor
from rocket_lander.models.actor_critic import ActorCriticNetwork
from rocket_lander.memory.rollout_buffer import RolloutBuffer



class PPO():

    def __init__(self, rede: ActorCriticNetwork, lr: list[float] = [3e-4, 3e-4, 1e-3], gamma: float = 0.99, lambda_gae: float = 0.65, epsilon_clip: float = 0.2, coef_entropia: float = 0.05, coef_valor: float = 1, epocas: int = 10, batch_size: int = 16) -> None:
        
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
        # self.optimizer = torch.optim.Adam(self.rede.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam([
            {'params': self.rede.backbone.parameters(), 'lr': lr[0]},
            {'params': self.rede.actor.parameters(), 'lr': lr[1]},
            {'params': self.rede.critic.parameters(), 'lr': lr[2]},
        ])

        self.perda = nn.MSELoss()

        self.recompensa_episodio = 0

    
    def calcula_vantagens(self, recompensas: torch.Tensor, valores: torch.Tensor, dones: torch.Tensor, proximo_valor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calcula a vantagem e o retorno dado uma ação. "mask" inverte o sentido de "done", onde 1 representa fim do episódio após ação, portanto sem próximo valor.

        Valor: Previsão do retorno esperado, feita pelo Critic.
        Vantagem: Diferença entre o retorno obtido e o esperado para um dado estado.
        Retorno: Recompensa cumulativa esperada. Soma total de todas as recompensas que o agente espera receber a partir de um momento específico até o final da tarefa, aplicando um fator de desconto para recompensas futuras.

        Args:
            recompensas (torch.Tensor): A recompença calculada ou aproximada por algo.
            valores (torch.Tensor): O valor de quão boa foi a ação tomada, vem do Critic.
            dones (torch.Tensor): Resposta da pergunta: "O episódio terminou (1) ou não (0) após a tomada da ação".
            proximo_valor (torch.Tensor): É o valor que entra no caso do cálculo do delta do último ponto, pois não há um próximo valor.

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

            dones_f = dones.float()
            mask = 1 - dones_f[t]
            delta = recompensas[t] + (self.gamma * mask * proximo_valor) - valores[t]
            gae = delta + (self.gamma * self.lambda_gae * mask * gae)
            vantagens[t] = gae
        
        retornos = vantagens + valores

        return vantagens, retornos

        # gae = torch.zeros_like(valores[0])

        # for t in reversed(range(len(recompensas))):
        #     next_value = proximo_valor if t == len(recompensas)-1 else valores[t+1]

        #     mask = 1.0 - dones[t].float()

        #     delta = recompensas[t] + self.gamma * next_value * mask - valores[t]
        #     gae = delta + self.gamma * self.lambda_gae * mask * gae
        #     vantagens[t] = gae

        # retornos = vantagens + valores
            
    def atualizar(self, buffer: RolloutBuffer, proximo_valor: torch.Tensor):
        """Diz como utilizar os dados armazenados para melhorar (atualizar) a política.

        Args:
            buffer (RolloutBuffer): O buffer da dados gerado pela classe RolloutBuffer, que armazena as informações dos estados.
            proximo_valor (torch.Tensor): É o valor que entra no caso do cálculo do delta do último ponto, pois não há um próximo valor.
        """

        dados = buffer.to_tensors()
        # Calcula vantagens e retornos
        vantagens, retornos = self.calcula_vantagens(recompensas=dados['recompensas'], valores=dados['valores'], dones=dados['dones'], proximo_valor=proximo_valor)
        vantagens = vantagens.detach()
        retornos = retornos.detach()

        # Normaliza as vantagens e retornos para melhorar a estabilidade de treinamento
        vantagens = (vantagens - vantagens.mean()) / (vantagens.std() + 1e-8)
        retornos = (retornos - retornos.mean()) / (retornos.std() + 1e-8)

        for e in range(self.epocas):
            indices = torch.randperm(buffer.size)
            for inicio in range(0, buffer.size, self.batch_size):
                fim = inicio + self.batch_size
                batch = indices[inicio:fim] # Divide em mini-batches

                # Extrair as informações do batch, com base na posição do buffer
                estados_batch = dados['estados'][batch]
                # print('estados_batch:', estados_batch[:5])
                acoes_batch = dados['acoes'][batch]
                log_probs_batch = dados['log_probs'][batch]
                vantagens_batch = vantagens[batch]
                retornos_batch = retornos[batch]

                # print("Estados:")
                # print(estados_batch)

                # x = estados_batch
                # for i, camada in enumerate(self.rede.backbone):
                #     x = camada(x)
                #     print(f"\nCamada {i}: {camada}")
                #     print("min:", x.min().item(),
                #         "max:", x.max().item(),
                #         "std:", x.std().item())
                #     print(x[:2])
                # print('-----')
            
                # Fazer o forward na rede e calcula as perdas
                with torch.no_grad():
                    feat = self.rede.backbone(estados_batch)

                # print('Época:', e)
                # print("Std estados :", estados_batch.std(dim=0))
                # print("Std features:", feat.std(dim=0))
                # print(feat[:5])
                # print("Feature std:", feat.std(dim=0).mean().item())
                loss, _, _, _ = self.calc_loss(estados=estados_batch, acoes=acoes_batch, log_probs_antigos=log_probs_batch, vantagens=vantagens_batch, retornos=retornos_batch)

                # Atualizar os parâmetros da rede
                self.optimizer.zero_grad()
                # print("Values:", dados[ 'valores'][:5].flatten().detach())
                # print("Retornos:", retornos_batch[:5].detach())
                loss.backward()
                # print("Grad Critic:", self.rede.critic[-1].weight.grad.norm().item())
                # print("Grad Actor:", self.rede.actor[-2].weight.grad.norm().item())
                self.optimizer.step()
                # print("Peso Critic:", self.rede.critic[-1].weight.norm().item())
                # print("Peso Actor :", self.rede.actor[-2].weight.norm().item())
                # print('')


        print('')
        print(f'Recompensas: mean: {torch.mean(dados['recompensas']).item():.3f}, std: {torch.std(dados['recompensas']).item():.3f}')
        print(f'Vantagens: mean: {vantagens.mean():.3f}, std: {vantagens.std():.3f}')
        print(f'Retornos: mean: {retornos.mean():.3f}, std: {retornos.std():.3f} $ Mínio e máximo: {retornos.min():.3f}, {retornos.max():.3f}')
        print(f'Valores: mean: {torch.mean(dados['valores']).item():.3f}, std: {torch.std(dados['valores']).item():.3f} $ Mínio e máximo: {torch.min(dados['valores']).item():.3f}, {torch.max(dados['valores']).item():.3f}')
        print('---')
        print("Std valores :", dados["valores"].std().item())
        print("Std retornos:", retornos.std().item())

        print('-'*41)
        print('### Atualização da política concluída ###')
        print('-'*41)

        # Limpar o buffer ao final de tudo
        self.recompensa_episodio += torch.sum(dados['recompensas']).item()
        buffer.clear()

    def calc_loss(self, estados: torch.Tensor, acoes: torch.Tensor, log_probs_antigos: torch.Tensor, vantagens: torch.Tensor, retornos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """A função de perda do PPO (rede), composta de três termos. "actor_loss" otimiza a política. "critic_loss" otimiza a estimativa de valor. "entropy" incentiva a exploração do espaço.

        Args:
            estados (torch.Tensor): Estados observados pelo agente.
            acoes (torch.Tensor): Ações tomadas durante a coleta de dados para o rollout.
            log_probs_antigos (torch.Tensor): Logaritmo da probabilidade das ações segundo a política antiga.
            vantagens (torch.Tensor): Estimativa da vantagem calculada pelo Generalized Advantage Estimation (GAE).
            retornos (torch.Tensor): Valor alvo para o treinamento do Crictic.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tupla com a perda total e a perda de cada parte do algoritmo, Actor, Critic e entropia.
        """

        # Obtém os novos valore do Critic e a nova distribuição de ações
        mu, valores = self.rede(estados)
        dist = Normal(mu, torch.exp(self.rede.log_std))

        # Encontra a probabilidade das ações segundo a nova política
        novo_log_prob = dist.log_prob(acoes).sum(dim=-1)
        razao = torch.exp(novo_log_prob - log_probs_antigos) # Compara com as probabilidades antigas

        # print('Razão: mean, std, min, max', razao.mean(),
        # razao.std(),
        # razao.min(),
        # razao.max())

        # print("ratio:", razao[:5])
        # print("adv:", vantagens[:5])
        # print("log_old:", log_probs_antigos[:5])
        # print("log_new:", novo_log_prob[:5])

        ##### Loss do Actor (deve ser maximizada)
        razao_clipada = torch.clamp(razao, 1-self.epsilon_clip, 1+self.epsilon_clip)
        actor_loss = -torch.min(razao * vantagens, razao_clipada * vantagens).mean()

        ##### Loss do Critic
        critic_loss = self.coef_valor * self.perda(valores, retornos)

        ##### Entropia - Favorecer a exploração (deve ser maximizada)
        entropia_loss = - self.coef_entropia * dist.entropy().sum(dim=-1).mean()

        ##### Combinar
        loss = actor_loss + critic_loss + entropia_loss

        print(f'## Loss Actor: {actor_loss:.3f}, Critic: {critic_loss:.3f}, Entropia: {entropia_loss:.3f}, Geral: {loss:.3f} ##')


        return loss, actor_loss, critic_loss, entropia_loss