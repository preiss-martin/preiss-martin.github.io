(function (g) {
  'use strict';
  function decode(s) {
    const b=Uint8Array.from(atob(s),c=>c.charCodeAt(0));
    const dv=new DataView(b.buffer); const a=new Float32Array(b.length/4);
    for(let i=0;i<a.length;i++) a[i]=dv.getFloat32(i*4,true);
    return a;
  }
  let weights;
  function convPool(a, n, ci, co, k, b) {
    const m=n-2, c=new Float32Array(m*m*co);
    for(let y=0;y<m;y++) for(let x=0;x<m;x++) {
      const t=(y*m+x)*co;
      c.set(b,t);
      for(let ky=0;ky<3;ky++) for(let kx=0;kx<3;kx++) {
        let ai=((y+ky)*n+x+kx)*ci, wi=(ky*3+kx)*ci*co;
        for(let i=0;i<ci;i++) {const v=a[ai+i]; for(let j=0;j<co;j++) c[t+j]+=v*k[wi+i*co+j];}
      }
    }
    const p=Math.floor(m/2), out=new Float32Array(p*p*co);
    for(let y=0;y<p;y++) for(let x=0;x<p;x++) for(let j=0;j<co;j++) {
      const u=((y*2)*m+x*2)*co+j;
      out[(y*p+x)*co+j]=Math.max(0,c[u],c[u+co],c[u+m*co],c[u+(m+1)*co]);
    }
    return out;
  }
  function predict(x) {
    if (x.length!==64*64*3) throw Error('Unexpected input shape');
    weights ||= g.CNN_WEIGHTS.map(decode);
    const w=weights;
    let z=convPool(x,64,3,32,w[0],w[1]);
    z=convPool(z,31,32,64,w[2],w[3]);
    z=convPool(z,14,64,128,w[4],w[5]);
    let logit=w[9][0];
    for(let j=0;j<64;j++) {let s=w[7][j]; for(let i=0;i<z.length;i++) s+=z[i]*w[6][i*64+j]; logit+=Math.max(0,s)*w[8][j];}
    return 1/(1+Math.exp(-logit));
  }
  g.SatelliteCNN={predict,decode};
})(globalThis);
