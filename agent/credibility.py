import tldextract
import yaml

from agent import config

# offline mode: use the public suffix list bundled with tldextract
_split_domain = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
_rules = None

SCHOLARLY_TYPES = {"article", "review", "book-chapter", "book"}


def load_rules():
    global _rules
    if _rules is None:
        _rules = yaml.safe_load(config.TRUST_FILE.read_text(encoding="utf-8"))
    return _rules


def find_tier(domain, suffix):
    tiers = load_rules()["tiers"]
    for tier in (1, 2, 4):
        if domain in tiers[tier]["domains"] or suffix in tiers[tier]["suffixes"]:
            return tier
    return 3


def score_source(url, author="", date="", work_type="", found_by=""):
    parts = _split_domain(url)
    domain = parts.top_domain_under_public_suffix or parts.domain
    tier = find_tier(domain, parts.suffix)
    reasons = [f"{domain} is tier {tier}"]

    # OpenAlex says what kind of scholarly work this is, which is a better signal than the doi.org link
    if found_by == "openalex" and tier == 3:
        if work_type in SCHOLARLY_TYPES:
            tier = 1
            reasons.append(f"scholarly {work_type} listed in OpenAlex")
        elif work_type == "preprint":
            tier = 2
            reasons.append("preprint listed in OpenAlex, not yet peer reviewed")

    rules = load_rules()
    weight = rules["tiers"][tier]["weight"]
    override = rules.get("overrides", {}).get(domain)
    if override:
        weight = override["weight"]
        reasons.append(override["reason"])

    if author:
        weight += 0.05
        reasons.append("author is named")
    if date:
        weight += 0.05
        reasons.append("publication date is given")

    # forums and social media stay low even with an author and date
    if tier == 4:
        weight = min(weight, 0.2)

    return {
        "domain": domain,
        "tier": tier,
        "label": rules["tiers"][tier]["label"],
        "weight": round(min(weight, 1.0), 2),
        "reasons": reasons,
    }
