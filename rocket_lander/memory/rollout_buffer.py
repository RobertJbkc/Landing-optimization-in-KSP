"""
Implementa um armazenador de dado que trabalha com retornos em forma de tensores.
Faz a coleta e aramzenamento dos dados observados.
"""

import torch


class RolloutBuffer():

    def __init__(self):
        self.buffer = {
            'estados': [],
            'acoes': [],
            'log_probs': [],
            'valores': [],
            'recompensas': [],
            'dones': [],
        }

    
    def add(self, estado: torch.Tensor, acao: torch.Tensor, log_prob: torch.Tensor, valor: torch.Tensor, recompensa: torch.Tensor, done: int):
        """Adiciona observações no buffer de dados via append nas listas do dicionário.

        Args:
            estado (torch.Tensor): Tensor que representa o estado observado.
            acao (torch.Tensor): Tensor contendo a ação executada.
            log_prob (torch.Tensor): Logaritmo da probabilidade de ação segundo a política vigente.
            valor (torch.Tensor): Estimativa do retorno (valor) V(s) produzida pelo Critic.
            recompensa (float): Recompensa recibida após executar a ação.
            done (int): Indica início e término de um episódio. "O episódio terminou (1) ou não (0) após essa ação?".
        """

        vetor_composto = (estado, acao, log_prob, valor, recompensa, done)
        for chave, variavel in zip(self.buffer, vetor_composto):
            self.buffer[chave].append(variavel)

    def clear(self):
        """Limpa o buffer de dados"""

        for chave in self.buffer:
            self.buffer[chave].clear()

    def to_tensors(self):
        """Transforma [state1, state2, state3, ...] em torch.stack(...). Seletivamente aplica "stack()" ou  "tensor()".
        Se o buffer de dados estiver vazio, o retorno é um dicionário vazio.

        Returns:
            dict: Dicionário convertido para torch.Tensor
        """

        dicionario = {}
        for chave, lista in self.buffer.items():
            if len(lista) == 0: # Tenho certeza de que se uma tiver comprimento nulo, as outras também o terão. Retorna "{}" por padrão
                return {}
            
            if all(isinstance(i, torch.Tensor) for i in lista):
                dicionario[chave] = torch.stack(lista)

            else:
                dicionario[chave] = torch.tensor(lista)

        return dicionario

    @property # Permite chamar o método como read-only --> como um parâmetro ou característicaVamos 
    def size(self):
        """Retorna, como read-only, o tamanho do tensor de estados"""

        return len(self.buffer['estados'])
