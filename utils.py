import json


def format_row(row):
    return {
        'point': {
            "lat": row["Lat"],
            "lng": row["Lon"]
        },
        "city": row["City"],
        "prob": row["score"],
        "label": row["Name"],
        "categories": [row["Kind"]],
        "xid": row["XID"],
        "photos": []
    }


def json_from_pandas_to_main_format(code: str):
    actual_list = json.loads(code)
    return list(map(format_row, actual_list))
