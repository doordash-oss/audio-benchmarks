// Copyright 2026 DoorDash, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const path=require('path');
const ROLE_A=path.join(__dirname,'..');
const {createDecisionTree}=require(path.join(ROLE_A,'decision_tree.js'));
const A=process.env.ARTIFACTS_PATH||path.join(__dirname, '..', '..', '..', 'artifacts');
const TID='order_up.Bella_Roma_Trattoria.24034958.g1.s1';
const NODE='g1_s1_i0_main';
let pass=0,fail=0;
function ck(n,got,want){const g=JSON.stringify(got),w=JSON.stringify(want);
  if(g===w){pass++;console.log('  PASS '+n);}else{fail++;console.log('  FAIL '+n+'\n        got '+g+'\n       want '+w);}}
function fresh(){
  const dt=createDecisionTree({treePath:path.join(A,'decision_trees',TID+'.json'),
    taskConfigPath:path.join(A,'order_up_tasks',TID+'.complex.task.json'),
    clipsBaseDirOverride:path.join(A,'clips',TID)});
  const cp=dt.parseClipId(NODE); dt.advanceTo(cp.stage,cp.item); dt.markPlayed(NODE);
  return dt;
}
// baseline: compound covers two items
const d0=fresh();
ck('compound leaves both items unacked', d0.getUnackedIds(), ['i0_main','i1_main']);

// generic alone
let d=fresh(); let r=d.recordAcks(['generic']);
ck('["generic"] acks both covered items', d.getUnackedIds(), []);
ck('["generic"] labels mapped_from=generic', [...new Set(r.accepted.map(a=>a.mapped_from))], ['generic']);

// target with an item
d=fresh(); r=d.recordAcks(['i0_main','explicit_question_target']);
ck('EXPLICIT question (item named) acks the item AND its peers', d.getUnackedIds(), []);
ck('peer ack is labelled explicit_question_target',
   r.accepted.filter(a=>a.mapped_from==='explicit_question_target').map(a=>a.short), ['i1_main']);
ck('explicit id keeps mapped_from=null',
   r.accepted.filter(a=>a.short==='i0_main').map(a=>a.mapped_from), [null]);

// target alone still expands
d=fresh(); d.recordAcks(['explicit_question_target']);
ck('sentinel alone still expands', d.getUnackedIds(), []);

// named item WITHOUT a sentinel must NOT ack the peer  <-- the rule that matters
d=fresh(); d.recordAcks(['i0_main']);
ck('IMPLICIT question (no item named) acks ONLY that item — peers stay unacked', d.getUnackedIds(), ['i1_main']);

// case-insensitivity + old spelling still works
d=fresh(); d.recordAcks(['EXPLICIT_QUESTION_TARGET']);
ck('sentinel is case-insensitive', d.getUnackedIds(), []);

// [] acks nothing
d=fresh(); d.recordAcks([]);
ck('[] acks nothing', d.getUnackedIds(), ['i0_main','i1_main']);

// unknown token rejected, no expansion
d=fresh(); r=d.recordAcks(['i9_main']);
ck('unknown id rejected', r.rejected.map(x=>x.reason), ['not_in_ackable_played']);
ck('unknown id causes no expansion', d.getUnackedIds(), ['i0_main','i1_main']);

// A list of explicit ids acks exactly those, with no expansion.
d=fresh(); r=d.recordAcks(['i0_main','i1_main']);
ck('two explicit ids ack both, no sentinel needed', d.getUnackedIds(), []);
ck('neither explicit id is labelled as an expansion',
   [...new Set(r.accepted.map(a=>a.mapped_from))], [null]);

// The retired spelling must NOT silently expand — a stale prompt would
// otherwise ack every item while looking like an unknown-id rejection.
d=fresh(); r=d.recordAcks(['i0_main','target']);
ck('retired "target" token does NOT expand', d.getUnackedIds(), ['i1_main']);
ck('retired "target" token is rejected', r.rejected.map(x=>x.reason), ['not_in_ackable_played']);

console.log('\n'+pass+' passed, '+fail+' failed');
process.exit(fail?1:0);
