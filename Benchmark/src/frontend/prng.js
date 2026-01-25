export class PRNG {
  constructor(seed = 0) {
    this.state = seed >>> 0;
  }

  next() {
    // Mulberry32
    this.state += 0x6d2b79f5;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  nextInt(max) {
    return Math.floor(this.next() * max);
  }

  pick(list) {
    return list[this.nextInt(list.length)];
  }
}
