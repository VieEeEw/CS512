from typing import List


def find_outlier(db: List[float], r: float = 2, pi: float = 0.2) -> None:
    card_db = len(db)
    for o in db:
        card_near = 0
        for o_prime in db:
            if abs(o - o_prime) <= r and o_prime != o:
                card_near += 1
        f_o = card_near / card_db
        print(f"$f({o})={f_o}", end="")
        if f_o <= pi:
            print(rf"\leq \pi$, {o} is an outlier.\\")
        else:
            print(rf"> \pi$, {o} is not an outlier.\\")


if __name__ == "__main__":
    D = [-4.5, -4, -3, -2.5, 0, 3, 3.5, 4, 4.5, 5]
    find_outlier(D)
