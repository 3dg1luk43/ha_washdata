# WashData Community Store

The WashData Community Store is a free, community-run library where WashData users share appliance setups -- programs, reference cycles, and optionally tuned detection settings -- so that anyone with the same appliance model can skip the "record and label from scratch" phase and start with accurate matching from day one.

The store lives at **[https://3dg1luk43.github.io/washdata-store](https://3dg1luk43.github.io/washdata-store)** and is built on GitHub Pages with a Firebase Firestore backend. The WashData panel has built-in integration so you can browse, adopt, and share without leaving Home Assistant.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting started -- enabling online features](#2-getting-started----enabling-online-features)
3. [Browsing and adopting a setup](#3-browsing-and-adopting-a-setup)
4. [Sharing your device](#4-sharing-your-device)
5. [Making a cycle shareable](#5-making-a-cycle-shareable)
6. [Profile matching and the community catalog](#6-profile-matching-and-the-community-catalog)
7. [Privacy and safety](#7-privacy-and-safety)
8. [Tips and best practices](#8-tips-and-best-practices)
9. [Store website](#9-store-website)

---

## 1. Introduction

Every appliance draws power differently -- even two machines of the same model can vary slightly depending on age, water pressure, load size, and local voltage. WashData learns your specific machine by watching its real cycles, but getting a useful profile library takes time: you run a program, record it, label it, and repeat for each program you care about.

The Community Store shortens that process considerably. If someone else with the same appliance model has already done the recording work and shared it, you can adopt their setup in seconds. The integration downloads their reference cycles and immediately has something to match against. You still own the data locally -- the store only seeds your starting point.

The catalog is organized as **Brand → Device → Program → Reference cycles**. Two users with the same washer model automatically land on the same device entry, so their contributions accumulate in one place.

Browsing and downloading are open to everyone with no account required. Contributing requires a free GitHub login, which is used only for attribution and spam prevention.

---

## 2. Getting started -- enabling online features

Online features in WashData are **disabled by default**. Nothing is sent to or fetched from the store until you explicitly opt in.

**To enable online features:**

1. Open the **WashData** panel from the Home Assistant sidebar.
2. Go to the **Advanced** tab.
3. Click the **gear icon** in the top area of the Advanced tab.
4. Toggle **Enable online features** on.
5. Use the **Brand** and **Model** pickers to declare which appliance you own.

The brand and model declaration is what links your WashData device to the correct store entry. Without it, the integration does not know which shared programs are relevant to you.

Once online features are enabled and your appliance model is declared, two new actions become available in the panel:

- **Browse community setups** -- shown on the onboarding card for a new device (no profiles yet) and accessible from the store section in the Advanced tab. Opens the store filtered to your declared model.
- **Share this device** -- available once you have programs with reference cycles to contribute.

You can disable online features at any time from the same gear menu. Disabling removes the store UI but does not delete any locally adopted cycles -- those remain part of your profile library.

---

## 3. Browsing and adopting a setup

If another user has shared a setup for your appliance model, adopting it takes less than a minute.

**Steps to adopt:**

1. Click **Browse community setups** (shown on the onboarding card on a new device, or from the Advanced tab).
2. The store panel opens, already filtered to programs contributed for your declared model.
3. Browse the available programs. Each shows program name, contributor, how many reference cycles are included, and a waveform preview.
4. Click **Download this setup** on a program or the full device setup you want.
5. A preview lists what will be imported: program names and the number of reference cycles for each.
6. An optional checkbox lets you also **adopt the contributor's detection and matching settings** -- the threshold and timing values they tuned for their machine. If your machine is the same model, this can save significant manual tuning. If you are unsure, leave it unchecked -- you can apply or review settings from the Settings tab later.
7. Confirm the import. Programs and cycles are added to your WashData device immediately.

After adopting, WashData has real reference cycles for each imported program. The next time your appliance runs one of those programs, the matcher compares the live power trace against those cycles and identifies the program automatically.

**You do not need to be logged in to browse or download.** Adopting a setup is a local import action -- it does not create any account or send any data to the store.

---

## 4. Sharing your device

Sharing contributes your recorded programs and reference cycles to the community so future users with the same appliance model get a head start.

**Before you can share**, you need at least one program with a reference cycle attached to it. See [Making a cycle shareable](#5-making-a-cycle-shareable) below.

**Steps to share:**

1. Open the WashData panel and confirm that online features are enabled and your brand/model are declared (see [Getting started](#2-getting-started----enabling-online-features)).
2. Click **Share this device**.
3. A share modal opens listing all your programs. Programs that have reference cycles are shown normally. Programs without reference cycles are shown dimmed with the note *"No reference cycles -- mark a cycle as ⭐ in the Cycles tab to include this profile"*. Only programs with reference cycles are included in the upload.
4. Optionally tick **Include phase maps** to bundle the phase ranges (Pre-wash, Heating, Spin, etc.) you have drawn for each program. Phase maps help other users get the phase-based time-remaining display working immediately after adopting.
5. Optionally tick **Include detection and matching settings** to bundle your tuned thresholds with the package. Only do this if you have meaningfully adjusted your thresholds from the defaults for a concrete reason -- for most machines the defaults are fine, and shipping aggressive thresholds can confuse users whose hardware is slightly different.
6. Read the consent notice and click **Share**.

Your package is uploaded in a **pending** state. Other users with the same model can see it and try it immediately, but it is marked as unreviewed. Once **five distinct users confirm** that the setup works for their machine, it is auto-approved and shown without the pending badge.

While your contribution is pending, you can edit its description or retract it from the store website. After approval you can still delete it if needed.

---

## 5. Making a cycle shareable

A cycle becomes a reference cycle -- eligible to be shared -- in one of two ways.

### Option A -- Mark a cycle as golden in the Cycles tab

1. Go to the **Cycles** tab in the WashData panel.
2. Find a completed cycle that ran the program you want to share. Look for a clean, uninterrupted run with a clear matched profile.
3. Open the cycle and click the **star icon (⭐)** to mark it as a golden reference cycle.

Golden cycles are used as the reference traces for their assigned profile. Mark at least one cycle per program you want to share. Two or three golden cycles per program give the matcher more to work with and make the shared package more useful.

A cycle must be assigned to a profile before it can be marked golden -- the star is greyed out on unlabelled cycles.

### Option B -- Record Mode

Record Mode captures the cleanest possible reference cycles by starting the recording before you press the button on the appliance, eliminating any startup noise.

1. Go to the **Overview** tab in the WashData panel.
2. In the **Manual Recording** widget, click **Start Recording**.
3. Start your appliance and run the program you want to record.
4. When the cycle finishes, click **Stop** in the widget.
5. Assign the recording to a profile in the **Profiles** tab.

Cycles captured via Record Mode are automatically marked as golden, so they are immediately ready to share.

---

## 6. Profile matching and the community catalog

When you adopt a setup from the store, WashData imports the reference cycles alongside the profile names. Those cycles become part of the local profile for that program on your device, exactly as if you had recorded them yourself.

The matcher -- which compares live power traces against stored reference cycles using shape correlation, duration, and energy -- treats adopted cycles identically to locally recorded ones. There is no separate "community" matching path. The integration always recomputes the matching envelope locally from the raw trace data; nothing from the store can influence how WashData talks to Home Assistant or controls your appliance.

Because every machine of the same model draws power slightly differently (due to age, water pressure, load, local voltage), an adopted setup may not immediately achieve top match confidence. This is normal. The matcher will improve as your own cycles accumulate alongside the adopted ones. You can help it along by:

- **Confirming correct matches** via the feedback attention card on the Overview tab. Each confirmation teaches the system the variance to expect on your specific unit.
- **Marking your own cycles as golden** to supplement the adopted reference cycles.
- **Using the Cleanup tab** in a profile to remove any adopted cycles that look like outliers on your machine -- hover a row to highlight its curve, tick it, and delete it.

The community catalog is additive. If you adopt a program and it works well straight away, you contribute nothing and that is perfectly fine. If you want to improve it for future users, recording a few clean cycles of your own and sharing them back is appreciated.

---

## 7. Privacy and safety

### Privacy

- **Browsing and downloading** require no account and transmit no personal information. Only the appliance type and model you declared are sent to filter the catalog.
- **Sharing** requires a GitHub login. The store stores only your GitHub public display name and avatar, shown as attribution on your contribution. No email address, home details, or other personal data are stored.
- **Shared cycle data** contains only the power trace -- watt values timestamped in seconds -- and the program name you assigned. Power traces carry no location information, no user identity, and no appliance serial numbers. There is nothing in a power trace that identifies you or your home.
- You can **delete your contributions** from the store website at any time, including after approval.

### Safety

- Adopted programs and cycles are imported locally into your WashData installation. They do not change your appliance's physical settings, do not connect to any external device, and do not send commands of any kind. They only affect how WashData recognises and labels your appliance's power cycles.
- Detection and matching settings adopted from the store are applied as option overrides exactly like any other settings change you make in the panel. You can review them in the **Settings** tab and revert to the defaults at any time.
- The integration fetches store data over a read-only API. There is no write path from the store back into your Home Assistant instance.

---

## 8. Tips and best practices

**Getting the best results from an adopted setup:**

- After adopting, run your two or three most common programs once each and confirm the matches in the feedback card on the Overview tab. A few confirmations calibrate the matcher to your specific unit quickly.
- If match confidence is consistently low for one program, record a clean cycle of it yourself using Record Mode and mark it as golden. This supplements the adopted reference cycles with data from your actual machine.
- If you adopted detection settings and something feels off (e.g. cycles ending too early or starting too late), review the Settings tab. The inline suggestions show what WashData recommends based on your observed cycles; the **Apply Suggestions** button can reset things to sensible values.

**Contributing high-quality setups:**

- Use Record Mode wherever possible. It produces cleaner reference cycles than retroactively marking a naturally-detected cycle.
- Provide 2-3 golden cycles per program, not just one. This gives the matcher a picture of normal variance and makes the shared package significantly more useful.
- Use clear, descriptive profile names -- e.g. "Cotton 60°C", "Eco 40°C", "Quick Wash 30 min" -- so other users can recognize their programs immediately.
- If your programs differ only in temperature or spin speed, create and share separate profiles for each variant. This lets users benefit from the finer matching granularity.
- Only tick **Include detection settings** if you have adjusted thresholds for a specific, concrete reason (e.g. your machine has an unusually long anti-wrinkle hold). Generic default-like settings add noise for users whose hardware behaves slightly differently.
- Only tick **Include phase maps** if you have actually drawn and verified the phase ranges in the profile editor. A half-drawn phase map is less useful than no phase map.

**Keeping your shared setup current:**

- Reference cycles are raw power traces, not version-dependent. They remain valid across WashData updates. But if you re-record a program after a significant upgrade and the new cycles look noticeably cleaner, uploading a refreshed share is appreciated.
- If you stop owning the appliance, consider deleting your contributions so the catalog stays accurate and maintainable.

---

## 9. Store website

The full WashData Community Store catalog is available at:

**[https://3dg1luk43.github.io/washdata-store](https://3dg1luk43.github.io/washdata-store)**

From the store website you can:
- Browse the full catalog by appliance brand and model, without an account.
- Preview shared programs and their power-cycle waveforms.
- Download individual reference cycles as JSON for manual import.
- Sign in with GitHub to contribute, rate, comment, and manage your own submissions.
