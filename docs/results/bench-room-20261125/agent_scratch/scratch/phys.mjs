import fs from 'node:fs';
const j=JSON.parse(fs.readFileSync('preview/physics.json','utf8'));
const only=process.argv[2];
for(const it of j.items){ if(only==='fail'&&it.pass) continue; const d=it.detail||{};console.log((it.pass?'ok  ':'FAIL')+' '+it.target.padEnd(17)+it.expect.padEnd(8)+(it.reason||'').padEnd(18)+Object.entries(d).map(([k,v])=>k+'='+v).join(' '))}
