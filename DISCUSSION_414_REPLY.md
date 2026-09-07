Fixed in 0.5.6.

You had it right, and there was a worse half you could not have seen: that sensor reported how many cycle *records* were stored, and WashData only keeps the most recent 200, so past 200 it would have frozen for good even if you never deleted anything.

It now reports the machine's running total, which WashData has been keeping correctly all along for the milestone notifications and simply never showed. It only rises, deleting a record does not touch it, it keeps counting past the storage limit, and it survives a wipe. The old number stays available as a `stored_cycles` attribute.

On ghost cycles: no prompt when you delete, you were right that it would be confusing. Obvious ghosts are already discarded before they are ever recorded, and for anything left the total is editable by hand in Advanced - Maintenance, like correcting a car's mileage. That also covers a machine that ran for years before WashData was installed.

Your report also exposed the same bug in WashData's own descale and filter reminders, which counted those same records. They now use the running total, and the Maintenance tab shows how close each task is instead of only telling you once it is due.

No need to coordinate with the Maintenance Supporter developer: they already ship a WashData setup, verified against our source, that reads this exact sensor as a lifetime counter at 30 / 50 / 100 cycles. It could not work while the sensor was not really one. It should now.
