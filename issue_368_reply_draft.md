# Draft reply - #368 Remote Start of Washers and Dryers

Copiable draft (no em dashes). Not posted.

---

Thanks for the really clear write-up, and for asking about direction up front rather than just building it. The use case makes sense and I know it is a common setup, so let me be straight with you about where I land.

I am going to decline autonomous remote/delayed start in WashData, for three reasons:

1. **Scope.** WashData is a passive power monitor. It watches a plug and interprets the power curve. It can cut and restore power today, but only as part of an explicit pause/resume that you initiate, and only immediately. Turning a heating appliance back on by itself, unattended, on a timer is a real safety and liability step-change from that, and I do not want WashData to own starting an appliance.

2. **Reliability.** This only works on machines that resume when power returns. Plenty of washers and dryers need a physical button press after power is restored, so a "remote start" feature would silently fail for a chunk of users, and I would rather not ship something that behaves that differently across hardware.

3. **It already belongs in the automation layer.** What you are describing is a scheduling job, and Home Assistant is very good at those. WashData already exposes the learned average and total cycle duration for each program, so an automation can work out when to switch the plug on and do it. You are effectively already doing this by hand, so this just formalizes it.

Concretely, something like this does the "finish by roughly X" part today (adjust entity names):

- Compute the power-on time as `target_finish_time - <learned average cycle duration>`.
- At that time, `switch.turn_on` your washer/dryer plug.

The average/total-duration values WashData publishes are what make that reliable rather than guesswork.

On the "intelligent" second phase (interrupting a running cycle to land exactly on a chosen finish time): I do not think that is deliverable reliably. Once a program is running, cutting and restoring power mid-cycle is not something I can make land on an exact minute across different machines and load types without risking a broken wash, so I would not want to build it.

One thing I am open to, if it would actually help you: I could expose a read-only "suggested power-on time to finish by X" as a sensor attribute or a service that just returns a timestamp, so your automation becomes trivial to write and always uses WashData's own learned durations. That keeps the actual switching in your hands (where the safety decision belongs) while removing the math. If that sounds useful, say so and I will look at adding it.

Appreciate the offer to help, and no worries at all about the HA-dev experience. If you want a hand turning the recipe above into a working automation for your specific entities, I am happy to sketch one.
