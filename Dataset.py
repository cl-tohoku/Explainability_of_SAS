from torch.utils import data


class hol_scoring_set(data.Dataset):

    def __init__(self, data_x, data_y, mask):
        self.data_x = data_x
        self.data_y = data_y
        self.mask = mask

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):

        x = self.data_x[index]
        y = self.data_y[index]
        mask = self.mask[index]

        return x, y, mask


class part_scoring_set(data.Dataset):
    def __init__(self, data_x, data_y, mask, attention, attention_flag=None):
        self.data_x = data_x
        self.data_y = data_y
        self.mask = mask
        self.attention = attention
        self.attention_flag = attention_flag

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):

        x = self.data_x[index]

        y = [y[index] for y in self.data_y]
        attention = [attn[index] for attn in self.attention]
        mask = self.mask[index]
        # flag = self.attention_flag[index] if self.attention_flag is not None else True
        flag = [flag[index] for flag in self.attention_flag] if self.attention_flag is not None else [
            True for _ in range(self.data_y)]

        return x, y, mask, attention, flag
