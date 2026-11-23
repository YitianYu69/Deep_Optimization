"""Dependency records for blocking failed model/feature paths before timing."""


def model_name(workload):
    kwargs=workload.get("factory_kwargs",{})
    if "build_timm_vit" in workload.get("factory",""):
        return kwargs.get("model_name","vit_base_patch16_224")
    return kwargs.get("model_name")


def blocking_reason(task,manifest):
    workload=task["workload"]
    for rule in manifest.get("blocked",[]):
        if rule.get("model") is not None and rule["model"]!=model_name(workload):continue
        if rule.get("case") is not None and rule["case"]!=task["case"]:continue
        if rule.get("batch_size") is not None and rule["batch_size"]!=workload.get("factory_kwargs",{}).get("batch_size"):continue
        if rule.get("execution") is not None and rule["execution"]!=task["settings"]["runtime"]["execution"]:continue
        if any(task["settings"]["runtime"].get(key)!=value for key,value in rule.get("runtime",{}).items()):continue
        if any(task["settings"]["fp8"].get(key)!=value for key,value in rule.get("fp8",{}).items()):continue
        return "Blocked by GPU preflight: "+rule["reason"]
    return None


def model_failure_rules(results, *, full_size=False):
    rules=[]
    for row in results:
        if row["status"]=="ok":continue
        task=row["task"]
        if not model_name(task["workload"]):continue
        if row.get("reason","").startswith("Blocked by GPU preflight:"):continue
        rule={"model":model_name(task["workload"]),"case":task["case"],
              "execution":task["settings"]["runtime"]["execution"],"reason":row.get("reason",row["status"]),
              "result_id":task["id"]}
        if full_size:rule["batch_size"]=task["workload"]["factory_kwargs"]["batch_size"]
        rules.append(rule)
    return rules
