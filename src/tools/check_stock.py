def check_stock(item_name: str) -> int:
    # giả lập: tìm trong dict tĩnh
    db = {"iPhone": 3, "AirPods": 10}
    return db.get(item_name, 0)