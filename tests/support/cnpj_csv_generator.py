"""Deterministic ``Estabelecimentos``-shaped CSV for the NFR-2 cost measurement."""

from __future__ import annotations

import argparse
import hashlib
import random
from collections.abc import Iterator
from pathlib import Path

DEFAULT_ROWS = 1_000_000
DEFAULT_SEED = 19
ENCODING = "iso-8859-1"
SEPARATOR = ";"

COLUMNS: tuple[str, ...] = (
    "cnpj_basico",
    "cnpj_ordem",
    "cnpj_dv",
    "identificador_matriz_filial",
    "nome_fantasia",
    "situacao_cadastral",
    "data_situacao_cadastral",
    "motivo_situacao_cadastral",
    "nome_cidade_exterior",
    "pais",
    "data_inicio_atividade",
    "cnae_fiscal_principal",
    "cnae_fiscal_secundaria",
    "tipo_logradouro",
    "logradouro",
    "numero",
    "complemento",
    "bairro",
    "cep",
    "uf",
    "municipio",
    "ddd_1",
    "telefone_1",
    "ddd_2",
    "telefone_2",
    "ddd_fax",
    "fax",
    "correio_eletronico",
    "situacao_especial",
    "data_situacao_especial",
)

_NAMES = (
    "PADARIA SÃO JOÃO",
    "CONFECÇÕES AÇAÍ",
    "MERCADINHO BOA VISTA",
    "OFICINA MECÂNICA IRMÃOS",
    "FARMÁCIA POPULAR",
    "",
)
_STREET_TYPES = ("RUA", "AVENIDA", "TRAVESSA", "RODOVIA", "PRAÇA")
_STREETS = ("DAS FLORES", "BRASIL", "GETÚLIO VARGAS", "JOSÉ BONIFÁCIO", "SETE DE SETEMBRO")
_DISTRICTS = ("CENTRO", "JARDIM AMÉRICA", "VILA NOVA", "SÃO CRISTÓVÃO")
_STATES = ("SP", "RJ", "MG", "BA", "RS", "PR", "PE", "CE", "DF", "AM")
_SITUATIONS = ("01", "02", "03", "04", "08")


def generate_rows(rows: int, seed: int) -> Iterator[tuple[str, ...]]:
    """Yield ``rows`` thirty-field tuples from one seeded generator."""
    rng = random.Random(seed)
    for index in range(rows):
        basico = f"{index:08d}"
        state = rng.choice(_STATES)
        secondary_count = rng.randrange(0, 3)
        secondary = ",".join(str(rng.randrange(1000000, 9999999)) for _ in range(secondary_count))
        yield (
            basico,
            f"{rng.randrange(1, 10):04d}",
            f"{rng.randrange(0, 100):02d}",
            str(rng.randrange(1, 3)),
            rng.choice(_NAMES),
            rng.choice(_SITUATIONS),
            _date(rng),
            f"{rng.randrange(0, 80):02d}",
            "",
            "",
            _date(rng),
            f"{rng.randrange(1000000, 9999999)}",
            secondary,
            rng.choice(_STREET_TYPES),
            rng.choice(_STREETS),
            str(rng.randrange(1, 5000)),
            "SALA " + str(rng.randrange(1, 300)) if rng.random() < 0.3 else "",
            rng.choice(_DISTRICTS),
            f"{rng.randrange(1000000, 99999999):08d}",
            state,
            f"{rng.randrange(1, 9999):04d}",
            f"{rng.randrange(11, 99)}",
            f"{rng.randrange(20000000, 99999999)}",
            "",
            "",
            "",
            "",
            f"contato{index}@exemplo.com.br" if rng.random() < 0.4 else "",
            "",
            "",
        )


def write_csv(path: Path, *, rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> str:
    """Write the file and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for record in generate_rows(rows, seed):
            line = SEPARATOR.join(f'"{value}"' for value in record) + "\n"
            encoded = line.encode(ENCODING)
            digest.update(encoded)
            handle.write(encoded)
    return digest.hexdigest()


def _date(rng: random.Random) -> str:
    return f"{rng.randrange(1970, 2026)}{rng.randrange(1, 13):02d}{rng.randrange(1, 29):02d}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    sha = write_csv(args.output, rows=args.rows, seed=args.seed)
    size = args.output.stat().st_size
    print(f"{args.output} rows={args.rows} seed={args.seed} bytes={size} sha256={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
