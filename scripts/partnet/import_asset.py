"""Import one authorized asset, validate, and bundle without using the CC0 catalog."""
import argparse
import json
from pathlib import Path
from gwm.compiler.partnet import import_single_hinge
from gwm.compiler.bundle import bundle
from gwm.compiler.validate import validate

p = argparse.ArgumentParser()
p.add_argument('archive')
p.add_argument('out')
p.add_argument('--sha256',required=True)
p.add_argument('--revision',required=True)
a = p.parse_args()
program, manifest, _ = import_single_hinge(a.archive,a.out,expected_sha256=a.sha256,revision=a.revision)
bundle(program,Path(a.out)/'game',manifest,validate(program))
limits=next(o for o in program['objects'] if o['id']=='lid')['motion']['range_deg']
html='''<!doctype html><meta charset="utf-8"><title>PartNet关节预览</title>
<style>body{font:16px system-ui;margin:12px;background:#eee}iframe{width:800px;height:600px;border:0}input{width:400px}</style>
<p>PartNet 单关节状态预览；拖动、松手保持和碰撞尚未验证。</p>
<label>关节状态 <input id="state" type="range" min="0" max="1" step="0.01" value="0"><output id="angle"></output></label><br>
<iframe src="index.html?headless=1&scene_only=1&w=800&h=600"></iframe>
<script>
const limits=LIMITS;
const frame=document.querySelector('iframe'),slider=document.querySelector('#state');
function update(){const t=Number(slider.value),g=frame.contentWindow.__game;g.seek(t);g.draw();document.querySelector('#angle').textContent=(limits[0]+t*(limits[1]-limits[0])).toFixed(1)+'°';}
slider.addEventListener('input',()=>{if(window.previewReady)update();});
const ready=setInterval(()=>{if(frame.contentWindow.__game?.readyState==='ready'){clearInterval(ready);window.previewReady=true;update();}},50);
</script>'''.replace('LIMITS',json.dumps(limits))
(Path(a.out)/'game/joint_preview.html').write_text(html,encoding='utf-8')
print('Validated and bundled',Path(a.out)/'game')
