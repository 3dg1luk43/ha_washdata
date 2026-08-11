# Draft replies for deferred / refused / needs-info issues (0.5.4)

Copiable drafts for the maintainer to post (or adapt). No em dashes. Nothing here has been posted.

---

## #368 - Remote start of washers/dryers  (recommend: REFUSE, with redirect)

Thanks for the detailed proposal. I have thought about this carefully and I do not think autonomous
remote start fits WashData, for a few reasons:

- WashData is a passive power monitor. It observes a plug and interprets the power curve. It can cut/
  restore power as an adjunct to an explicit user pause/resume, but autonomously energizing a heating
  appliance on a timer is a real safety and liability step-change from that, and I do not want to own it.
- It only works on the subset of appliances that resume when power returns. Many require a physical
  button press after power is restored, so the feature would be unreliable across the user base.
- The finish-by use case is already fully achievable in Home Assistant's automation layer, which is
  where it belongs: WashData already exposes the learned average/total cycle duration, so an automation
  can compute the power-on time and switch the plug on itself.

Concretely, this recipe does what you want today (adjust to your entities):

    trigger: time-based, fire at (target_finish_time - <learned average duration>)
    action: switch.turn_on: switch.washer_plug

If it would help, I am open to exposing a read-only "suggested power-on time to finish by X" as a sensor
attribute so the automation is trivial to write, leaving the actual switching to you. Let me know if that
would be useful and I will consider it. But the autonomous scheduler/interrupt itself I am going to
decline.

---

## #344 - Import historical power data (CSV / recorder)  (NEEDS-INFO, then phased build)

I like this and I want to build it. The engine pieces already exist (WashData can read recorder history
and can replay a trace through the real detector), so the new work is a CSV/recorder ingest, splitting a
continuous multi-month trace into candidate cycles, and a review screen to label them. Two things before I
start:

1. Could you share the exact column format of your CSV export (header row + a few sample lines)? I want
   the importer to match what Home Assistant's history export actually produces.
2. Heads-up on a limitation we confirmed in this thread: Home Assistant keeps full-resolution history for
   only ~10 days by default, then downsamples to hourly. Older-than-10-day data will not detect reliably.
   The CSV path (your own full-resolution export) is therefore the broadly useful one, so I will prioritize
   that; recorder-history import will mainly help users who raised their retention.

Plan once I have the schema: CSV ingest + a background segmentation pass (chunked so it never freezes),
then a review/label screen, with imported cycles landing as reference cycles so they sharpen matching
without touching your usage/energy stats.

---

## #364 - Smart termination splits a wash at a shorter prefix profile (cf #288)  (NEEDS-INFO)

Thanks, this is a clear and well-documented report, and it is a real regression of the #288 guarantee. I
have traced why the #288 prefix guard misses your two cases (the untrained-longer-program case cannot be
caught by a profile-based guard, and for the trained case the partial trace scores below the guard's shape
threshold against the full envelope). Fixing it safely means changing detection behaviour, so I want to
reproduce it exactly and A/B any fix before shipping.

Could you attach the cycle export for case 2 (the Baumwolle 60 run, cycle `25ebafe290f7`)? Advanced ->
Diagnostics -> export that cycle. With that I can reproduce the mis-termination and validate a fix does not
reintroduce the late-finish behaviour earlier issues fixed.

---

## #334 - Water-level program variants indistinguishable  (accepted; NEEDS-INFO to ship)

Your source reading is accurate, and this is on the list. One correction on the fix location: adding a
fill role to phase_match only affects the time-remaining estimate, not which program is chosen, so it will
not fix the mislabel you saw. The right place is the Stage-5 group member-picker, and it has to be gated
on whether fill is actually separable on your machine: on real front-loaders fill is often noise and a
fill term makes matching worse, whereas on a clean top-loader fill signal it helps a lot.

To validate the gate on your hardware before I ship it, could you export a couple of the mismatched cycles
(the full-water "Heavy Duty Regular" and the "Medium Water Level" variant) via Advanced -> Diagnostics? I
want to confirm the fill window is a clean discriminator on your machine and not just in synthetic data.

---

## #353 - Consume an external programme value from a smart appliance  (accepted in principle; held)

This fits WashData well (it is still passive: read another entity, feed the existing program override), and
about 70% of the plumbing already exists. I am holding it briefly to design the per-device value -> profile
mapping properly, since the program strings differ across integrations (Home Connect, etc.) and the mapping
UI is the part that needs care. I will follow up with a design.

---

## #352 - "Summer break" notice

Not an issue; can be closed.
