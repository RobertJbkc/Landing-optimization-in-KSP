import torch


class RolloutBuffer():
    """
    Esta classe deve, APENAS:
    - Armazenar transições
    - Limpar o buffer
    - Devolver os dados em forma de tensor
    """

    def __init__(self):
        # self.estados = []
        # self.acoes = []
        # self.log_probs = []
        # self.valores = []
        # self.recompensas = []
        # self.dones = []

        self.buffer = {
            'estados': [],
            'acoes': [],
            'log_probs': [],
            'valores': [],
            'recompensas': [],
            'dones': [],
        }

    
    def add(self, estado: torch.Tensor, acao: torch.Tensor, log_prob: torch.Tensor, valor: torch.Tensor, recompensa: float, done: int):
        """Adiciona observações no buffer de dados via append nas listas do dicionário.

        Args:
            estado (torch.Tensor): Tensor que representa o estado observado.
            acao (torch.Tensor): Tensor contendo a ação executada.
            log_prob (torch.Tensor): Logaritmo da probabilidade de ação segundo a política.
            valor (torch.Tensor): Estimativa V(s) produzida pelo Critic.
            recompensa (float): Recompensa recibida após executar a ação.
            done (int): Indica início e término de um episódio. "O episódio terminou (1) ou não (0) após essa ação?".
        """

        ### Pode ser vaálido criar uma etapa de normalização de entradas
        # O as_tensor já realiza a verificação se é ou não um tensor. O condicional não deve ser necessário.
        # if not isinstance(recompensa, float):
        #     recompensa = float(recompensa)

        # if not isinstance(done, bool):
        #     done = bool(done)

        # if not isinstance(estado, torch.Tensor):
        #     estado = torch.as_tensor(estado)

        # if not isinstance(acao, torch.Tensor):
        #     acao = torch.as_tensor(acao)

        # if not isinstance(log_prob, torch.Tensor):
        #     log_prob = torch.as_tensor(log_prob)

        # if not isinstance(valor, torch.Tensor):
        #     valor = torch.as_tensor(valor)

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
            
            if isinstance(lista[0], torch.Tensor):
                dicionario[chave] = torch.stack(lista)

            else:
                dicionario[chave] = torch.tensor(lista)

        return dicionario

    @property # Permite chamar o método como read-only --> como um parâmetro ou característicaVamos 
    def size(self):
        """Retorna, como read-only, o tamanho do tensor de estados"""

        return len(self.buffer['estados'])
