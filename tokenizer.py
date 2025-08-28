class CustomTokenizer:
    def __init__(self):
        # Danh sách ký tự thường gặp trong SMILES
        self.charset = ['C', 'O', 'N', 'S', '(', ')', '=', '#', '[', ']', '+', '-', '@', 'H', '1', '2', '3', '4', '5',
                        '6', '7', '8', '9', 'B', 'r', 'F', 'I', 'P', '/', '\\', '.', '%', '*', 'c', 'n', 'o', 's', 'Cl', 'Br', 'Si']
        self.token2idx = {ch: idx + 1 for idx, ch in enumerate(self.charset)}  # index từ 1
        self.unk_token = len(self.charset) + 1  # token cho ký tự lạ

    def encode(self, smiles: str):
        tokens = []
        i = 0
        while i < len(smiles):
            # Ưu tiên token 2 ký tự như "Cl", "Br"
            if i + 1 < len(smiles) and smiles[i:i+2] in self.token2idx:
                tokens.append(self.token2idx[smiles[i:i+2]])
                i += 2
            else:
                tokens.append(self.token2idx.get(smiles[i], self.unk_token))  # unknown → UNK token
                i += 1
        return tokens
