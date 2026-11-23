#!/usr/bin/env python3
"""Convert a ComfyUI UI workflow (with subgraphs) to API prompt format.

usage: ui2api.py workflow.json object_info.json > prompt.json
Handles: nested subgraphs, Reroute, PrimitiveNode, bypassed (mode 4) and
muted (mode 2) nodes, Note/MarkdownNote.
"""
import json
import sys

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
SKIP = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}


def norm_links(links):
    out = {}
    for l in links:
        if isinstance(l, dict):
            out[l["id"]] = (l["origin_id"], l["origin_slot"], l["target_id"], l["target_slot"], l.get("type"))
        else:
            out[l[0]] = (l[1], l[2], l[3], l[4], l[5])
    return out


class Scope:
    def __init__(self, nodes, links, prefix, parent=None, instance=None):
        self.nodes = {n["id"]: n for n in nodes}
        self.links = norm_links(links)
        self.prefix = prefix
        self.parent = parent          # parent Scope
        self.instance = instance      # subgraph instance node in parent

    def fid(self, nid):
        return f"{self.prefix}{nid}"


class Converter:
    def __init__(self, wf, info):
        self.info = info
        self.defs = {sg["id"]: sg for sg in wf.get("definitions", {}).get("subgraphs", [])}
        self.prompt = {}
        self.root = Scope(wf["nodes"], wf["links"], "")

    def widget_names(self, ntype):
        return self._widgets(self.info[ntype]["input"], "")

    def _widgets(self, spec, prefix, values=None):
        """Ordered widget names. Dynamic combos expand using the chosen key,
        which is only known from values -> return generator-like list of
        (name, kind) where kind 'dyn' needs a value lookup; handled in map_widgets."""
        names = []
        for sect in ("required", "optional"):
            for name, cfg in spec.get(sect, {}).items():
                t = cfg[0]
                opts = cfg[1] if len(cfg) > 1 and isinstance(cfg[1], dict) else {}
                full = prefix + name
                if t == "COMFY_DYNAMICCOMBO_V3":
                    names.append((full, opts["options"]))
                    continue
                is_widget = (isinstance(t, list) or (isinstance(t, str) and t in WIDGET_TYPES)
                             or "widgetType" in opts)
                if opts.get("forceInput"):
                    is_widget = False
                if is_widget:
                    names.append((full, None))
                    if opts.get("control_after_generate"):
                        names.append((None, None))
                    if opts.get("image_upload"):
                        names.append((None, None))
        return names

    def map_widgets(self, spec, wv, prefix=""):
        out, i = {}, 0
        for name, dyn in self._widgets(spec, prefix):
            if i >= len(wv):
                break
            v = wv[i]; i += 1
            if name is None:
                continue
            out[name] = v
            if dyn is not None:
                opt = next((o for o in dyn if o["key"] == v), None)
                if opt:
                    sub, used = self.map_widgets(opt["inputs"], wv[i:], name + ".")
                    out.update(sub); i += used
        return out, i

    # ---- resolve a link inside a scope to an API value ([id, slot] or literal)
    def resolve(self, scope, link_id):
        o_id, o_slot, _, _, ltype = scope.links[link_id]
        return self.resolve_origin(scope, o_id, o_slot, ltype)

    def resolve_origin(self, scope, o_id, o_slot, ltype=None):
        if o_id == -10:  # subgraph input -> parent
            return self.instance_input(scope, o_slot)
        node = scope.nodes[o_id]
        t = node["type"]
        if t == "Reroute":
            return self.resolve(scope, node["inputs"][0]["link"])
        if t == "PrimitiveNode":
            return node["widgets_values"][0]
        if node.get("mode") == 4:  # bypass: pass through first input of same type
            otype = node["outputs"][o_slot]["type"]
            for inp in node.get("inputs", []):
                if inp.get("link") is not None and inp["type"] == otype:
                    return self.resolve(scope, inp["link"])
            return None
        if t in self.defs:  # output of a subgraph instance
            sub = self.sub_scope(scope, node)
            sg = self.defs[t]
            for lid, (so, ss, st, sts, _) in sub.links.items():
                if st == -20 and sts == o_slot:
                    return self.resolve(sub, lid)
            raise KeyError(f"subgraph {t} output {o_slot} unresolved")
        return [scope.fid(o_id), o_slot]

    def instance_input(self, scope, slot):
        inst, parent = scope.instance, scope.parent
        sg = self.defs[inst["type"]]
        name = sg["inputs"][slot]["name"]
        for inp in inst.get("inputs", []):
            if inp["name"] == name and inp.get("link") is not None:
                return self.resolve(parent, inp["link"])
        wv = inst.get("widgets_values") or []
        widx = [i["name"] for i in sg["inputs"]
                if all(x in WIDGET_TYPES for x in str(i["type"]).split(","))]
        if len(widx) != len(wv):
            print(f"WARN subgraph {inst['id']} widget count {len(widx)} vs {len(wv)}", file=sys.stderr)
        if name in widx and widx.index(name) < len(wv):
            return wv[widx.index(name)]
        return None

    _scopes = {}

    def sub_scope(self, scope, inst):
        key = scope.fid(inst["id"])
        if key not in self._scopes:
            sg = self.defs[inst["type"]]
            self._scopes[key] = Scope(sg["nodes"], sg["links"], key + ":", scope, inst)
        return self._scopes[key]

    def emit_scope(self, scope):
        for nid, node in scope.nodes.items():
            t = node["type"]
            if t in SKIP or node.get("mode") in (2, 4):
                continue
            if t in self.defs:
                self.emit_scope(self.sub_scope(scope, node))
                continue
            if t not in self.info:
                print(f"WARN unknown node type {t} ({scope.fid(nid)})", file=sys.stderr)
                continue
            inputs = {}
            wv = node.get("widgets_values") or []
            if isinstance(wv, dict):
                inputs.update({k: v for k, v in wv.items() if k != "videopreview"})
            else:
                mapped, used = self.map_widgets(self.info[t]["input"], wv)
                if used != len(wv):
                    print(f"WARN {t} {scope.fid(nid)} used {used}/{len(wv)} widget values", file=sys.stderr)
                inputs.update(mapped)
            for inp in node.get("inputs", []):
                if inp.get("link") is not None:
                    val = self.resolve(scope, inp["link"])
                    if val is not None:
                        inputs[inp["name"]] = val
            self.prompt[scope.fid(nid)] = {"class_type": t, "inputs": inputs,
                                           "_meta": {"title": node.get("title", t)}}

    def run(self):
        self.emit_scope(self.root)
        # prune nodes whose link inputs point to missing nodes
        changed = True
        while changed:
            changed = False
            for k in list(self.prompt):
                for v in self.prompt[k]["inputs"].values():
                    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] not in self.prompt:
                        print(f"WARN pruning {k} ({self.prompt[k]['class_type']}): missing {v[0]}", file=sys.stderr)
                        del self.prompt[k]
                        changed = True
                        break
        return self.prompt


if __name__ == "__main__":
    wf = json.load(open(sys.argv[1]))
    info = json.load(open(sys.argv[2]))
    json.dump(Converter(wf, info).run(), sys.stdout, indent=1)
