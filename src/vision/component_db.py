"""Map YOLO class names to canonical component types."""

CANONICAL_TYPES = frozenset(
    {
        "ArduinoUNO",
        "Breadboard",
        "LED",
        "Resistor",
        "JumperWire",
        "Button",
    }
)

# Common dataset label aliases -> canonical
ALIASES: dict[str, str] = {
    "arduino": "ArduinoUNO",
    "arduino_uno": "ArduinoUNO",
    "uno": "ArduinoUNO",
    "breadboard": "Breadboard",
    "led": "LED",
    "resistor": "Resistor",
    "jumper": "JumperWire",
    "jumper_wire": "JumperWire",
    "wire": "JumperWire",
    "button": "Button",
    "push_button": "Button",
}


DISPLAY_ZH: dict[str, str] = {
    "ArduinoUNO": "Arduino UNO",
    "Breadboard": "面包板",
    "LED": "LED",
    "Resistor": "电阻",
    "JumperWire": "杜邦线",
    "Button": "按键",
}


def display_name(type_name: str) -> str:
    return DISPLAY_ZH.get(type_name, type_name)


def normalize_class_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in ALIASES:
        return ALIASES[key]
    for part in name.replace("-", " ").split():
        p = part.lower()
        if p in ALIASES:
            return ALIASES[p]
    # Title-case fallback if already canonical-ish
    titled = "".join(w.capitalize() for w in name.replace("_", " ").split())
    if titled in CANONICAL_TYPES:
        return titled
    if name in CANONICAL_TYPES:
        return name
    return name
