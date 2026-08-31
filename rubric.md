# Triage rubric — v0.1 (DRAFT — this one is mine, replace it with yours)

> This is a placeholder so the classifier runs. It has not been derived from
> data and it should not survive contact with your gold set. After labelling,
> read your own `my_rationale` column back and rewrite this in your words.
> Bump the version when you do. Every eval run records which version produced it.

## Scope
Customer support tickets. "Priority" here means commercial and customer
urgency — not ITIL service impact. Do not import severity language from
incident management.

## Levels

**high** — the customer is blocked, money or data is at risk, or the issue
is time-bound with a deadline the customer names. Also: repeated contact
about the same unresolved problem.

**medium** — the customer is impeded but has a workaround, or the request
needs action but names no deadline.

**low** — information requests, general enquiries, feedback, and anything
where no action is currently blocked.

## Refusal conditions
- If the ticket is too short or vague to identify what the customer needs,
  return your best guess with confidence below 0.4 and say so in the rationale.
- Politeness, formality and length are not severity signals. A courteous
  message can be urgent; an abrupt one can be trivial.

## Known trap
The dataset's own labels disagree with this rubric fairly often. That is
expected and is not a reason to change the rubric — record the disagreement.
