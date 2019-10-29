from torch.utils.data import Dataset, DataLoader
import os
import json

class JokesDataset(Dataset):
    def __init__(self, jokes_dataset_path = 'jokes_dataset/'):
        super().__init__()

        reddit_jokes_path = os.path.join(jokes_dataset_path, 'reddit_jokes.json')

        with open(reddit_jokes_path) as f:
            data = json.load(f)

        self.joke_list = []
        self.end_of_text_token = "<|endoftext|>"

        for idx, joke_json in enumerate(data):
            joke_str = f"{joke_json['title']} {joke_json['body']}{self.end_of_text_token}"
            self.joke_list.append(joke_str)

    def __len__(self):
        return len(self.joke_list)

    def __getitem__(self, item):
        return self.joke_list[item]

dataset = JokesDataset()
print(dataset.__len__())

joke_loader = DataLoader(dataset, batch_size=1, shuffle=True)
