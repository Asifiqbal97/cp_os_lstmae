"""Maps CICIoMT2024 Attack_type subtype strings to attack families.

Numeric suffixes (ICMP1, ICMP2, ...) are file-split artifacts, not
distinct subtypes -- stripped before matching.
"""

FAMILY_PREFIXES = {
    "ARP_Spoofing": "Spoofing",
    "MQTT-DDoS": "MQTT",
    "MQTT-DoS": "MQTT",
    "MQTT-Malformed": "MQTT",
    "Recon-": "Recon",
    "TCP_IP-DDoS": "DDoS",
    "TCP_IP-DoS": "DoS",
}

BENIGN_LABEL = "Benign"
KNOWN_FAMILIES = ["Spoofing", "MQTT", "Recon", "DDoS", "DoS"]


def subtype_to_family(attack_type: str) -> str:
    """Map a raw Attack_type value to its family. Raises on unmapped values
    rather than silently defaulting -- an unmapped subtype means the taxonomy
    is incomplete and must be fixed, not swallowed.
    """
    if attack_type == BENIGN_LABEL:
        return BENIGN_LABEL

    for prefix, family in FAMILY_PREFIXES.items():
        if attack_type.startswith(prefix):
            return family

    raise ValueError(
        f"Unmapped Attack_type '{attack_type}' -- add it to FAMILY_PREFIXES "
        f"in family_map.py"
    )
