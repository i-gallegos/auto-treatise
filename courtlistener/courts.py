"""Court identity from a CourtListener court slug: name, abbreviation, level,
jurisdiction and circuit.
"""

from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

COURT_LEVELS = (
    "scotus",  # Supreme Court of the United States
    "federal_appellate",  # courts of appeals, incl. D.C. and Federal Circuits
    "federal_district",  # the 94 district courts
    "federal_bankruptcy",  # bankruptcy courts and bankruptcy appellate panels
    "federal_specialized",  # Article I / specialized Article III courts
    "state_high",  # a state's court of last resort
    "state_appellate",  # intermediate appellate
    "state_trial",  # trial courts of record
    "territorial",  # territorial courts that are not Article III districts
    "tribal",
    "other",
)

#: Rank within one court system only. Higher binds lower.
AUTHORITY_RANK: dict[str, int] = {
    "scotus": 4,
    "federal_appellate": 3,
    "federal_district": 2,
    "federal_bankruptcy": 1,
    "federal_specialized": 2,
    "state_high": 3,
    "state_appellate": 2,
    "state_trial": 1,
}

_FEDERAL_LEVELS = frozenset(
    {
        "scotus",
        "federal_appellate",
        "federal_district",
        "federal_bankruptcy",
        "federal_specialized",
    }
)
_STATE_LEVELS = frozenset({"state_high", "state_appellate", "state_trial"})


def court_system(level: str | None) -> str | None:
    """``"federal"``, ``"state"`` or ``None``: the hierarchy ``level`` is in."""
    if level in _FEDERAL_LEVELS:
        return "federal"
    if level in _STATE_LEVELS:
        return "state"
    return None


# --------------------------------------------------------------------------- #
# Federal appellate courts
# --------------------------------------------------------------------------- #

#: CourtListener slug -> (circuit key, Bluebook abbreviation, full name).
FEDERAL_APPELLATE: dict[str, tuple[str, str, str]] = {
    "ca1": (
        "1",
        "1st Cir.",
        "United States Court of Appeals for the First Circuit",
    ),
    "ca2": (
        "2",
        "2d Cir.",
        "United States Court of Appeals for the Second Circuit",
    ),
    "ca3": (
        "3",
        "3d Cir.",
        "United States Court of Appeals for the Third Circuit",
    ),
    "ca4": (
        "4",
        "4th Cir.",
        "United States Court of Appeals for the Fourth Circuit",
    ),
    "ca5": (
        "5",
        "5th Cir.",
        "United States Court of Appeals for the Fifth Circuit",
    ),
    "ca6": (
        "6",
        "6th Cir.",
        "United States Court of Appeals for the Sixth Circuit",
    ),
    "ca7": (
        "7",
        "7th Cir.",
        "United States Court of Appeals for the Seventh Circuit",
    ),
    "ca8": (
        "8",
        "8th Cir.",
        "United States Court of Appeals for the Eighth Circuit",
    ),
    "ca9": (
        "9",
        "9th Cir.",
        "United States Court of Appeals for the Ninth Circuit",
    ),
    "ca10": (
        "10",
        "10th Cir.",
        "United States Court of Appeals for the Tenth Circuit",
    ),
    "ca11": (
        "11",
        "11th Cir.",
        "United States Court of Appeals for the Eleventh Circuit",
    ),
    "cadc": (
        "dc",
        "D.C. Cir.",
        "United States Court of Appeals for the District of Columbia Circuit",
    ),
    "cafc": (
        "fed",
        "Fed. Cir.",
        "United States Court of Appeals for the Federal Circuit",
    ),
}


# --------------------------------------------------------------------------- #
# Federal district courts
# --------------------------------------------------------------------------- #

#: The 94 district courts: CourtListener slug -> USPS code of its territory.
#: Not parsed from the slug: ``med`` is Maine, ``ned`` is Nebraska, ``nmid`` is the
#: Northern Mariana Islands.
FEDERAL_DISTRICT_STATE: dict[str, str] = {
    # 1st Circuit
    "med": "ME",
    "mad": "MA",
    "nhd": "NH",
    "rid": "RI",
    "prd": "PR",
    # 2d Circuit
    "ctd": "CT",
    "nyed": "NY",
    "nynd": "NY",
    "nysd": "NY",
    "nywd": "NY",
    "vtd": "VT",
    # 3d Circuit
    "ded": "DE",
    "njd": "NJ",
    "paed": "PA",
    "pamd": "PA",
    "pawd": "PA",
    "vid": "VI",
    # 4th Circuit
    "mdd": "MD",
    "nced": "NC",
    "ncmd": "NC",
    "ncwd": "NC",
    "scd": "SC",
    "vaed": "VA",
    "vawd": "VA",
    "wvnd": "WV",
    "wvsd": "WV",
    # 5th Circuit
    "laed": "LA",
    "lamd": "LA",
    "lawd": "LA",
    "msnd": "MS",
    "mssd": "MS",
    "txed": "TX",
    "txnd": "TX",
    "txsd": "TX",
    "txwd": "TX",
    # 6th Circuit
    "kyed": "KY",
    "kywd": "KY",
    "mied": "MI",
    "miwd": "MI",
    "ohnd": "OH",
    "ohsd": "OH",
    "tned": "TN",
    "tnmd": "TN",
    "tnwd": "TN",
    # 7th Circuit
    "ilcd": "IL",
    "ilnd": "IL",
    "ilsd": "IL",
    "innd": "IN",
    "insd": "IN",
    "wied": "WI",
    "wiwd": "WI",
    # 8th Circuit
    "ared": "AR",
    "arwd": "AR",
    "iand": "IA",
    "iasd": "IA",
    "mnd": "MN",
    "moed": "MO",
    "mowd": "MO",
    "ned": "NE",
    "ndd": "ND",
    "sdd": "SD",
    # 9th Circuit
    "akd": "AK",
    "azd": "AZ",
    "cacd": "CA",
    "caed": "CA",
    "cand": "CA",
    "casd": "CA",
    "hid": "HI",
    "idd": "ID",
    "mtd": "MT",
    "nvd": "NV",
    "ord": "OR",
    "waed": "WA",
    "wawd": "WA",
    "gud": "GU",
    "nmid": "MP",
    # 10th Circuit
    "cod": "CO",
    "ksd": "KS",
    "nmd": "NM",
    "oked": "OK",
    "oknd": "OK",
    "okwd": "OK",
    "utd": "UT",
    "wyd": "WY",
    # 11th Circuit
    "alnd": "AL",
    "almd": "AL",
    "alsd": "AL",
    "flmd": "FL",
    "flnd": "FL",
    "flsd": "FL",
    "gamd": "GA",
    "gand": "GA",
    "gasd": "GA",
    # D.C. Circuit
    "dcd": "DC",
}

#: USPS code -> circuit key. Territories sit in the circuit that hears their
#: appeals.
STATE_CIRCUIT: dict[str, str] = {
    # 1st
    "ME": "1",
    "MA": "1",
    "NH": "1",
    "RI": "1",
    "PR": "1",
    # 2d
    "CT": "2",
    "NY": "2",
    "VT": "2",
    # 3d
    "DE": "3",
    "NJ": "3",
    "PA": "3",
    "VI": "3",
    # 4th
    "MD": "4",
    "NC": "4",
    "SC": "4",
    "VA": "4",
    "WV": "4",
    # 5th
    "LA": "5",
    "MS": "5",
    "TX": "5",
    # 6th
    "KY": "6",
    "MI": "6",
    "OH": "6",
    "TN": "6",
    # 7th
    "IL": "7",
    "IN": "7",
    "WI": "7",
    # 8th
    "AR": "8",
    "IA": "8",
    "MN": "8",
    "MO": "8",
    "NE": "8",
    "ND": "8",
    "SD": "8",
    # 9th
    "AK": "9",
    "AZ": "9",
    "CA": "9",
    "HI": "9",
    "ID": "9",
    "MT": "9",
    "NV": "9",
    "OR": "9",
    "WA": "9",
    "GU": "9",
    "MP": "9",
    # 10th
    "CO": "10",
    "KS": "10",
    "NM": "10",
    "OK": "10",
    "UT": "10",
    "WY": "10",
    # 11th
    "AL": "11",
    "FL": "11",
    "GA": "11",
    # D.C.
    "DC": "dc",
}


# --------------------------------------------------------------------------- #
# Specialized federal courts
# --------------------------------------------------------------------------- #

#: Slug -> (level, abbreviation, full name). Nationwide jurisdiction.
FEDERAL_SPECIALIZED: dict[str, tuple[str, str, str]] = {
    "cit": (
        "federal_specialized",
        "Ct. Int'l Trade",
        "United States Court of International Trade",
    ),
    "uscfc": (
        "federal_specialized",
        "Fed. Cl.",
        "United States Court of Federal Claims",
    ),
    "tax": ("federal_specialized", "T.C.", "United States Tax Court"),
    "cavc": (
        "federal_specialized",
        "Vet. App.",
        "United States Court of Appeals for Veterans Claims",
    ),
    "armfor": (
        "federal_specialized",
        "C.A.A.F.",
        "United States Court of Appeals for the Armed Forces",
    ),
    "acca": (
        "federal_specialized",
        "A. Ct. Crim. App.",
        "United States Army Court of Criminal Appeals",
    ),
    "afcca": (
        "federal_specialized",
        "A.F. Ct. Crim. App.",
        "United States Air Force Court of Criminal Appeals",
    ),
    "nmcca": (
        "federal_specialized",
        "N-M. Ct. Crim. App.",
        "United States Navy-Marine Corps Court of Criminal Appeals",
    ),
    "cgcca": (
        "federal_specialized",
        "C.G. Ct. Crim. App.",
        "United States Coast Guard Court of Criminal Appeals",
    ),
    "jpml": (
        "federal_specialized",
        "J.P.M.L.",
        "United States Judicial Panel on Multidistrict Litigation",
    ),
    "fisc": (
        "federal_specialized",
        "FISA Ct.",
        "United States Foreign Intelligence Surveillance Court",
    ),
    "fiscr": (
        "federal_specialized",
        "FISA Ct. Rev.",
        "United States Foreign Intelligence Surveillance Court of Review",
    ),
    "mspb": (
        "federal_specialized",
        "M.S.P.B.",
        "United States Merit Systems Protection Board",
    ),
    "bia": ("federal_specialized", "B.I.A.", "Board of Immigration Appeals"),
    # CourtListener also carries older/longer slugs for some of the same courts.
    "uscgcoca": (
        "federal_specialized",
        "C.G. Ct. Crim. App.",
        "United States Coast Guard Court of Criminal Appeals",
    ),
}

#: Federal courts that are not in the 94-district table: historical districts
#: that no longer exist, and CourtListener's generic catch-all slug.
#: slug -> (USPS or None, abbreviation, full name)
FEDERAL_DISTRICT_EXTRA: dict[str, tuple[str | None, str, str]] = {
    # Abolished in 1978 and split into the Central and Southern Districts, but
    # it still appears on pre-1978 opinions.
    "illinoised": (
        "IL",
        "E.D. Ill.",
        "United States District Court for the Eastern District of Illinois",
    ),
    # No district identified. Level is knowable, geography is not.
    "usdistct": (None, None, "United States District Court"),
}

#: Executive-branch opinion writers CourtListener files as courts. Level
#: ``other``, which ``AUTHORITY_RANK`` does not rank.
NON_COURTS: dict[str, tuple[str | None, str, str]] = {
    "texag": (
        "TX",
        "Tex. Att'y Gen. Op.",
        "Office of the Attorney General of Texas",
    ),
    "olc": ("US", "O.L.C.", "Office of Legal Counsel"),
}

#: Bankruptcy appellate panels: slug -> circuit key.
BAP_CIRCUIT: dict[str, str] = {
    "bap1": "1",
    "bap9": "9",
    "bap10": "10",
    "bapme": "1",
}


# --------------------------------------------------------------------------- #
# State courts of last resort
# --------------------------------------------------------------------------- #

#: Courts of last resort: slug -> USPS code. Texas and Oklahoma have two each.
STATE_HIGH_COURT: dict[str, str] = {
    "ala": "AL",
    "alaska": "AK",
    "ariz": "AZ",
    "ark": "AR",
    "cal": "CA",
    "colo": "CO",
    "conn": "CT",
    "del": "DE",
    "dc": "DC",
    "fla": "FL",
    "ga": "GA",
    "haw": "HI",
    "idaho": "ID",
    "ill": "IL",
    "ind": "IN",
    "iowa": "IA",
    "kan": "KS",
    "ky": "KY",
    "la": "LA",
    "me": "ME",
    "md": "MD",
    "mass": "MA",
    "mich": "MI",
    "minn": "MN",
    "miss": "MS",
    "mo": "MO",
    "mont": "MT",
    "neb": "NE",
    "nev": "NV",
    "nh": "NH",
    "nj": "NJ",
    "nm": "NM",
    "ny": "NY",
    "nc": "NC",
    "nd": "ND",
    "ohio": "OH",
    "okla": "OK",
    "oklacrimapp": "OK",
    "or": "OR",
    "pa": "PA",
    "ri": "RI",
    "sc": "SC",
    "sd": "SD",
    "tenn": "TN",
    "tex": "TX",
    "texcrimapp": "TX",
    "utah": "UT",
    "vt": "VT",
    "va": "VA",
    "wash": "WA",
    "wva": "WV",
    "wis": "WI",
    "wyo": "WY",
    # Territorial courts of last resort.
    "prsupreme": "PR",
    "guam": "GU",
    "vi": "VI",
    "nmariana": "MP",
}

#: Non-federal courts the slug cues below misclassify or cannot reach.
#: slug -> (level, USPS, abbreviation, full name or None). Territorial systems
#: use the ``state_*`` levels.
STATE_COURT_EXTRA: dict[str, tuple[str, str, str, str | None]] = {
    "prapp": (
        "state_appellate",
        "PR",
        "P.R. App.",
        "Court of Appeals of Puerto Rico",
    ),
    "pacommwct": (
        "state_appellate",
        "PA",
        "Pa. Commw. Ct.",
        "Commonwealth Court of Pennsylvania",
    ),
    "pasuperct": (
        "state_appellate",
        "PA",
        "Pa. Super. Ct.",
        "Superior Court of Pennsylvania",
    ),
    # New York's trial court of general jurisdiction is called the Supreme
    # Court; the Appellate Division (``nyappdiv``) sits above it and the Court
    # of Appeals (``ny``) above that.
    "nysupct": (
        "state_trial",
        "NY",
        "N.Y. Sup. Ct.",
        "Supreme Court of New York",
    ),
    "superctguam": (
        "state_trial",
        "GU",
        "Guam Super. Ct.",
        "Superior Court of Guam",
    ),
    "amsamoa": (
        "state_high",
        "AS",
        "High Ct. Am. Samoa",
        "High Court of American Samoa",
    ),
}

#: Slug families by shape: (pattern, level, USPS, abbreviation, full name or None)
_SLUG_FAMILIES: tuple[
    tuple[re.Pattern[str], str, str, str, str | None], ...
] = (
    (re.compile(r"^txctapp\d+$"), "state_appellate", "TX", "Tex. App.", None),
    (
        re.compile(r"^nysupct[a-z]+$"),
        "state_trial",
        "NY",
        "N.Y. Sup. Ct.",
        None,
    ),
    (
        re.compile(r"^nycrimct[a-z]+$"),
        "state_trial",
        "NY",
        "N.Y. Crim. Ct.",
        None,
    ),
    (
        re.compile(r"^nycityct[a-z]*$"),
        "state_trial",
        "NY",
        "N.Y. City Ct.",
        None,
    ),
    (
        re.compile(r"^pactcompl[a-z]+$"),
        "state_trial",
        "PA",
        "Pa. Ct. Com. Pl.",
        None,
    ),
)

_STATE_PREFIXES = sorted(
    {slug for slug in STATE_HIGH_COURT}, key=len, reverse=True
)

# Level cues for state courts not in a table, searched anywhere in the slug.
# Appellate is tested first: ``calctapp`` and ``flaapp`` also contain "dist".
_STATE_LEVEL_CUES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(ctapp|ctcrimapp|crimapp|appdiv|appct|ctciv|civapp|agrapp|appter)"
        ),
        "state_appellate",
    ),
    (re.compile(r"app"), "state_appellate"),
    (
        re.compile(
            r"(superct|super|circuit|cir|distct|dist|county|cty|municipal|justice|probate|famct|juv|chan|ch)"
        ),
        "state_trial",
    ),
)

#: A CourtListener court slug is lowercase alphanumeric (``ca9``, ``nysd``).
_SLUG_RE = re.compile(r"^[a-z0-9]+$")


# --------------------------------------------------------------------------- #
# Slug -> full record
# --------------------------------------------------------------------------- #


def _blank() -> dict[str, Any]:
    return {
        "court_id": None,
        "court_name": None,
        "court_abbreviation": None,
        "court_level": None,
        "jurisdiction": None,
        "circuit": None,
    }


def normalize_court(court_id: str | None) -> dict[str, Any]:
    """Court record for a CourtListener court slug. Fields are ``None`` when
    the slug is not recognized."""
    record = _blank()
    if not court_id:
        return record
    slug = court_id.strip().lower()
    if not _SLUG_RE.match(slug):
        return record
    record["court_id"] = slug

    if slug == "scotus":
        record.update(
            court_name="Supreme Court of the United States",
            court_abbreviation="U.S.",
            court_level="scotus",
            jurisdiction="US",
        )
        return record

    if slug in FEDERAL_APPELLATE:
        circuit, abbrev, name = FEDERAL_APPELLATE[slug]
        record.update(
            court_name=name,
            court_abbreviation=abbrev,
            court_level="federal_appellate",
            jurisdiction="US" if circuit == "fed" else None,
            circuit=circuit,
        )
        return record

    if slug in FEDERAL_DISTRICT_STATE:
        usps = FEDERAL_DISTRICT_STATE[slug]
        record.update(
            court_name=_district_name(slug, usps),
            court_abbreviation=_district_abbrev(slug, usps),
            court_level="federal_district",
            jurisdiction=usps,
            circuit=STATE_CIRCUIT.get(usps),
        )
        return record

    if slug in BAP_CIRCUIT:
        circuit = BAP_CIRCUIT[slug]
        record.update(
            court_name=f"Bankruptcy Appellate Panel of the {_ordinal(circuit)} Circuit",
            court_abbreviation=f"B.A.P. {_ordinal(circuit)} Cir.",
            court_level="federal_bankruptcy",
            circuit=circuit,
        )
        return record

    # Bankruptcy courts: the district slug with a trailing 'b'.
    if slug.endswith("b") and f"{slug[:-1]}d" in FEDERAL_DISTRICT_STATE:
        usps = FEDERAL_DISTRICT_STATE[f"{slug[:-1]}d"]
        record.update(
            court_name=_district_name(f"{slug[:-1]}d", usps).replace(
                "District Court", "Bankruptcy Court"
            ),
            court_abbreviation=f"Bankr. {_district_abbrev(f'{slug[:-1]}d', usps)}",
            court_level="federal_bankruptcy",
            jurisdiction=usps,
            circuit=STATE_CIRCUIT.get(usps),
        )
        return record

    if slug in FEDERAL_SPECIALIZED:
        level, abbrev, name = FEDERAL_SPECIALIZED[slug]
        record.update(
            court_name=name,
            court_abbreviation=abbrev,
            court_level=level,
            jurisdiction="US",
        )
        return record

    if slug in FEDERAL_DISTRICT_EXTRA:
        usps, abbrev, name = FEDERAL_DISTRICT_EXTRA[slug]
        record.update(
            court_name=name,
            court_abbreviation=abbrev,
            court_level="federal_district",
            jurisdiction=usps,
            circuit=STATE_CIRCUIT.get(usps) if usps else None,
        )
        return record

    if slug in NON_COURTS:
        usps, abbrev, name = NON_COURTS[slug]
        record.update(
            court_name=name,
            court_abbreviation=abbrev,
            court_level="other",
            jurisdiction=usps,
        )
        return record

    if slug in STATE_COURT_EXTRA:
        level, usps, abbrev, name = STATE_COURT_EXTRA[slug]
        record.update(
            court_name=name,
            court_abbreviation=abbrev,
            court_level=level,
            jurisdiction=usps,
        )
        return record

    for pattern, level, usps, abbrev, name in _SLUG_FAMILIES:
        if pattern.match(slug):
            record.update(
                court_name=name,
                court_abbreviation=abbrev,
                court_level=level,
                jurisdiction=usps,
            )
            return record

    if slug in STATE_HIGH_COURT:
        usps = STATE_HIGH_COURT[slug]
        record.update(
            court_abbreviation=_STATE_ABBREV.get(usps),
            court_level="state_high",
            jurisdiction=usps,
        )
        return record

    # Not tabulated: guess the level from a slug cue.
    prefix = next((p for p in _STATE_PREFIXES if slug.startswith(p)), None)
    if prefix:
        for pattern, level in _STATE_LEVEL_CUES:
            if pattern.search(slug):
                record.update(
                    court_level=level,
                    jurisdiction=STATE_HIGH_COURT[prefix],
                )
                return record
        record.update(jurisdiction=STATE_HIGH_COURT[prefix])
    return record


_DIVISIONS = {
    "n": "Northern",
    "s": "Southern",
    "e": "Eastern",
    "w": "Western",
    "m": "Middle",
    "c": "Central",
}

_DIVISION_ABBREV = {
    "Northern": "N.",
    "Southern": "S.",
    "Eastern": "E.",
    "Western": "W.",
    "Middle": "M.",
    "Central": "C.",
}

_STATE_NAMES = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "the District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "GU": "Guam",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MP": "the Northern Mariana Islands",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "PR": "Puerto Rico",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "VI": "the Virgin Islands",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "AS": "American Samoa",
}

_STATE_ABBREV = {
    "AL": "Ala.",
    "AK": "Alaska",
    "AZ": "Ariz.",
    "AR": "Ark.",
    "CA": "Cal.",
    "CO": "Colo.",
    "CT": "Conn.",
    "DE": "Del.",
    "DC": "D.C.",
    "FL": "Fla.",
    "GA": "Ga.",
    "GU": "Guam",
    "HI": "Haw.",
    "ID": "Idaho",
    "IL": "Ill.",
    "IN": "Ind.",
    "IA": "Iowa",
    "KS": "Kan.",
    "KY": "Ky.",
    "LA": "La.",
    "ME": "Me.",
    "MD": "Md.",
    "MA": "Mass.",
    "MI": "Mich.",
    "MN": "Minn.",
    "MS": "Miss.",
    "MO": "Mo.",
    "MP": "N. Mar. I.",
    "MT": "Mont.",
    "NE": "Neb.",
    "NV": "Nev.",
    "NH": "N.H.",
    "NJ": "N.J.",
    "NM": "N.M.",
    "NY": "N.Y.",
    "NC": "N.C.",
    "ND": "N.D.",
    "OH": "Ohio",
    "OK": "Okla.",
    "OR": "Or.",
    "PA": "Pa.",
    "PR": "P.R.",
    "RI": "R.I.",
    "SC": "S.C.",
    "SD": "S.D.",
    "TN": "Tenn.",
    "TX": "Tex.",
    "UT": "Utah",
    "VT": "Vt.",
    "VA": "Va.",
    "VI": "V.I.",
    "WA": "Wash.",
    "WV": "W. Va.",
    "WI": "Wis.",
    "WY": "Wyo.",
    "AS": "Am. Samoa",
}


def _division(slug: str, usps: str) -> str | None:
    """The division of a district slug, if it has one: ``nysd`` -> Southern."""
    body = slug[:-1]  # drop the trailing 'd'
    if slug == "nmid":  # Northern Mariana Islands, not a division
        return None
    if len(body) == 3 and body[:2] == usps.lower():
        return _DIVISIONS.get(body[2])
    return None


def _district_name(slug: str, usps: str) -> str:
    state = _STATE_NAMES.get(usps, usps)
    division = _division(slug, usps)
    where = (
        f"the {division} District of {state}"
        if division
        else f"the District of {state}"
    )
    if usps == "DC":
        where = "the District of Columbia"
    return f"United States District Court for {where}"


_ALL_INITIALS = re.compile(r"^(?:[A-Z]\.)+$")


def _district_abbrev(slug: str, usps: str) -> str:
    """Bluebook district abbreviation: ``S.D.N.Y.``, ``E.D. Mich.``, ``D. Me.``

    The space rule is the fiddly part: initials run together (``S.D.N.Y.``,
    ``D.N.J.``) but a word-abbreviation takes a space (``E.D. Mich.``,
    ``D. Mass.``). Keyed off whether the state abbreviation is entirely
    single-letter initials.
    """
    state = _STATE_ABBREV.get(usps, usps)
    division = _division(slug, usps)
    if usps == "DC":
        return "D.D.C."
    joiner = "" if _ALL_INITIALS.match(state) else " "
    if division:
        return f"{_DIVISION_ABBREV[division]}D.{joiner}{state}"
    return f"D.{joiner}{state}"


def _ordinal(circuit: str) -> str:
    named = {"dc": "D.C.", "fed": "Federal"}
    if circuit in named:
        return named[circuit]
    suffix = {1: "st", 2: "d", 3: "d"}.get(int(circuit), "th")
    return f"{circuit}{suffix}"


def binds(superior: dict[str, Any], inferior: dict[str, Any]) -> bool:
    """Whether ``superior`` is vertically binding on ``inferior``.

    Only the uncontroversial relations, and only inside one hierarchy:

    * the Supreme Court binds every federal court;
    * a court of appeals binds the district and bankruptcy courts of its
      circuit;
    * a state's court of last resort binds that state's lower courts.

    Everything else is ``False``, including the interesting cases — a sister
    circuit, a state court on federal law, a federal court on state law. Those
    are persuasive-authority questions, and encoding a guess here would let a
    hierarchy-aware tree quietly assert something the doctrine does not.
    """
    sup_system = court_system(superior.get("court_level"))
    inf_system = court_system(inferior.get("court_level"))
    if sup_system is None or sup_system != inf_system:
        return False
    if superior.get("court_level") == "scotus":
        return inferior.get("court_level") != "scotus"
    if sup_system == "federal":
        if superior.get("court_level") != "federal_appellate":
            return False
        if inferior.get("court_level") not in (
            "federal_district",
            "federal_bankruptcy",
        ):
            return False
        return bool(superior.get("circuit")) and superior[
            "circuit"
        ] == inferior.get("circuit")
    if superior.get("court_level") != "state_high":
        return False
    if inferior.get("court_level") not in ("state_appellate", "state_trial"):
        return False
    return bool(superior.get("jurisdiction")) and superior[
        "jurisdiction"
    ] == inferior.get("jurisdiction")
