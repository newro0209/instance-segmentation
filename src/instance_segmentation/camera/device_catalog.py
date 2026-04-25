def infer_device_type_from_name(device_name: str) -> str:
    lowered_name = device_name.lower()
    if any(keyword in lowered_name for keyword in ["virtual", "obs", "ndi", "manycam", "camtwist"]):
        return "VIRTUAL"
    return "PHYSICAL"


def build_summary_rows(
    pnp_devices: list[dict[str, str]],
    openable_indices: list[int],
) -> list[dict[str, str]]:
    return [
        {
            "Index": str(openable_indices[index]) if index < len(openable_indices) else "N/A",
            "System Name": device_info["FriendlyName"],
            "Device Type": infer_device_type_from_name(device_info["FriendlyName"]),
            "Class": device_info["Class"],
            "Status": device_info["Status"],
        }
        for index, device_info in enumerate(pnp_devices)
    ]


def render_horizontal_table(summary_rows: list[dict[str, str]]) -> list[str]:
    if not summary_rows:
        return ["No camera devices found."]

    column_order = [
        "Index",
        "System Name",
        "Device Type",
        "Class",
        "Status",
    ]

    column_widths = {
        column: max(len(column), max(len(str(row[column])) for row in summary_rows))
        for column in column_order
    }

    header = " | ".join(f"{column:<{column_widths[column]}}" for column in column_order)
    separator = "-+-".join("-" * column_widths[column] for column in column_order)
    rows = [
        " | ".join(f"{str(row[column]):<{column_widths[column]}}" for column in column_order)
        for row in summary_rows
    ]

    return [header, separator, *rows]
