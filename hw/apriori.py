from typing import List, Dict, Set

MIN_SUP = 1264
CSV_FILE = "purchase_history.csv"
CSV_FILE_DEBUG = "purchase_history_debug.csv"


def read_csv(fname: str, split=False) -> List[Set[str]]:
    with open(fname, "r") as f:
        return [{k.strip().split("-")[-1] if split else k.strip() for k in line.strip().split(",")} for line in f]


class Apriori:
    def __init__(self, min_sup: int = MIN_SUP, debug=False):
        self.tx = read_csv(CSV_FILE_DEBUG if debug else CSV_FILE)
        self.k = 1
        self.min_sup = min_sup
        temp: Dict[str, int] = {}
        for row in self.tx:
            for word in row:
                temp[word] = temp.setdefault(word, 0) + 1
        self.item_set: List[Set[str]] = []
        for item, count in temp.items():
            if count >= min_sup:
                self.item_set.append({item})
                print(f"{item}: {count}")

    def mine(self):
        while len(self.item_set) > 0:
            self.k += 1
            self.gen_candidate()

    def gen_candidate(self):
        temp: Dict[str, int] = {}
        for i in range(len(self.item_set) - 1):
            for j in range(i, len(self.item_set)):
                diff = self.item_set[i].difference(self.item_set[j])
                if len(diff) == 1:
                    serialized = ",".join(sorted(self.item_set[j].union(diff)))
                    temp[serialized] = temp.setdefault(serialized, 0) + 1
        target = self.k * (self.k - 1) / 2
        self.scan_and_update([s for s, count in temp.items() if count >= target])

    def scan_and_update(self, candidates: List[str]):
        temp: Dict[str, int] = {}
        for row in self.tx:
            for candi in candidates:
                item = set(candi.split(","))
                if item.issubset(row):
                    temp[candi] = temp.setdefault(candi, 0) + 1
        self.item_set.clear()
        for candi, count in temp.items():
            if count >= self.min_sup:
                print(f"{candi}: {count}")
                self.item_set.append(set(candi.split(",")))


if __name__ == "__main__":
    apriori = Apriori(debug=True, min_sup=2)
    apriori.mine()
