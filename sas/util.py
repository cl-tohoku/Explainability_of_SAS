def get_item_num(item, main_factors):
    assert item in main_factors, f"{item} is not in {main_factors}"
    # CRLEAのでーたに「A_2」などがあるのでいったんコメントアウト
    # assert len(
    #     item) == 1 and 'A' <= item and 'Z' >= item, f"{item} is an invalid item name"
    # num = ord(item) - ord('A')
    num = main_factors.index(item)
    return num
