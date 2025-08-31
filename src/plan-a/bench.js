// Minimal bench harness for Plan A simulation runs
// This file is browser-side and intended to be used by the demo runner.

export class Bench {
    constructor() {
        this.frameTimes = [];
        this.startTime = null;
        this.last = null;
    }

    startFrame() {
        this.last = performance.now();
        if (this.startTime === null) this.startTime = this.last;
    }

    endFrame() {
        const now = performance.now();
        const dt = now - this.last;
        this.frameTimes.push(dt);
        if (this.frameTimes.length > 1000) this.frameTimes.shift();
    }

    getStats() {
        if (!this.frameTimes.length) return null;
        const arr = this.frameTimes.slice();
        arr.sort((a,b)=>a-b);
        const mean = arr.reduce((s,v)=>s+v,0)/arr.length;
        const median = arr[Math.floor(arr.length/2)];
        const p90 = arr[Math.floor(arr.length*0.9)];
        return { mean, median, p90, samples: arr.length };
    }

    toCSVLine(info = {}) {
        const s = this.getStats() || {mean:0,median:0,p90:0,samples:0};
        const parts = [new Date().toISOString(), s.mean.toFixed(3), s.median.toFixed(3), s.p90.toFixed(3), s.samples];
        for (const k of Object.keys(info)) parts.push(info[k]);
        return parts.join(',');
    }
}
