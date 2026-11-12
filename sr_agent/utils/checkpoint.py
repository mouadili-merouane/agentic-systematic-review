"""Persistance robuste aux coupures de session : JSON par étape + JSONL incrémental + piste d'audit."""
import json, pathlib, datetime, hashlib


class RunStore:
    def __init__(self, run_dir):
        self.dir = pathlib.Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    # --- résultats d'étape complets ---
    def path(self, name):
        return self.dir / name

    def has(self, name):
        return self.path(name).exists()

    def save_json(self, name, obj):
        tmp = self.path(name + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path(name))  # écriture atomique : jamais de fichier à moitié écrit

    def load_json(self, name, default=None):
        return json.loads(self.path(name).read_text(encoding="utf-8")) if self.has(name) else default

    def delete(self, name):
        if self.has(name):
            self.path(name).unlink()

    # --- traitement incrémental (screening, extraction...) ---
    def append_jsonl(self, name, rec):
        with self.path(name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def read_jsonl(self, name):
        if not self.has(name):
            return []
        return [json.loads(l) for l in self.path(name).read_text(encoding="utf-8").splitlines() if l.strip()]

    def done_ids(self, name, key="id"):
        return {r[key] for r in self.read_jsonl(name)}

    # --- piste d'audit (transparence / reproductibilité) ---
    def log(self, agent, action, **details):
        self.append_jsonl("audit_log.jsonl", {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "agent": agent, "action": action, **details})

    # --- compteurs PRISMA ---
    def prisma(self, **counts):
        cur = self.load_json("prisma_counts.json", {})
        cur.update(counts)
        self.save_json("prisma_counts.json", cur)
        return cur


def stable_id(*parts):
    return hashlib.sha1("|".join(str(p).lower().strip() for p in parts).encode()).hexdigest()[:12]
