"use client";
import {useState} from "react";
const choices={Gender:["Male","Female"],Married:["Yes","No"],Dependents:["0","1","2","3+"],Education:["Graduate","Not Graduate"],Self_Employed:["No","Yes"],Credit_History:["1","0"],Property_Area:["Urban","Semiurban","Rural"]};
const numbers=["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"];
const initial={Gender:"Male",Married:"Yes",Dependents:"0",Education:"Graduate",Self_Employed:"No",Credit_History:"1",Property_Area:"Urban",ApplicantIncome:"5000",CoapplicantIncome:"1500",LoanAmount:"120",Loan_Amount_Term:"360",model:"tuned"};
const names={Gender:"Gender",Married:"Married",Dependents:"Dependents",Education:"Education",Self_Employed:"Self-employed",Credit_History:"Credit history code (0 or 1)",Property_Area:"Property area",ApplicantIncome:"Applicant income (dataset units)",CoapplicantIncome:"Co-applicant income (dataset units)",LoanAmount:"Loan amount (dataset units)",Loan_Amount_Term:"Loan term (dataset units)"};
export default function Home(){
 const [form,setForm]=useState(initial),[result,setResult]=useState(null),[busy,setBusy]=useState(false),[error,setError]=useState("");
 function change(key,value){setForm(previous=>({...previous,[key]:value}));setResult(null);setError("");}
 async function submit(event){event.preventDefault();setBusy(true);setError("");setResult(null);
  try{const payload={...form};for(const key of [...numbers,"Credit_History"])payload[key]=form[key]===""?null:Number(form[key]);for(const key of Object.keys(choices))if(form[key]==="")payload[key]=null;
   const base=(process.env.NEXT_PUBLIC_API_URL||"http://localhost:8000").replace(/\/$/,"");
   const response=await fetch(`${base}/predict`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
   const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==="string"?data.detail:JSON.stringify(data.detail));setResult(data);
  }catch(e){setError(e.message||"Prediction failed. Check the backend URL and CORS settings.");}finally{setBusy(false);}}
 return <main><header><span className="badge">CAPSTONE-05 · K-NEAREST NEIGHBORS</span><h1>LoanLens</h1><p>Explore a model's predicted loan approval from applicant details.</p></header>
 <aside>Educational demonstration only. Not a real approval, credit assessment, or financial advice. This model has not been audited for fairness or validated for lending.</aside>
 <div className="layout"><section className="card"><h2>Applicant details</h2><p className="muted">Unknown values are imputed using training data. Units are not documented in the supplied CSV; use its original scale.</p>
 <form onSubmit={submit}><div className="grid">{Object.entries(choices).map(([key,options])=><label key={key}>{names[key]}<select value={form[key]} onChange={e=>change(key,e.target.value)}><option value="">Unknown</option>{options.map(value=><option key={value} value={value}>{value}</option>)}</select></label>)}
 {numbers.map(key=><label key={key}>{names[key]}<input type="number" min={key==="LoanAmount"||key==="Loan_Amount_Term"?"0.01":"0"} step="any" value={form[key]} onChange={e=>change(key,e.target.value)} placeholder="Unknown"/></label>)}
 <label>Model<select value={form.model} onChange={e=>change("model",e.target.value)}><option value="default">Default KNN (k=5)</option><option value="tuned">CV-tuned KNN</option></select></label></div>
 <button disabled={busy} type="submit">{busy?"Predicting…":"Predict loan status"}</button></form></section>
 <section className="card output" aria-live="polite"><h2>Prediction result</h2>{error&&<p className="error">{error}</p>}{result?<><div className={result.label==="Approved"?"approved":"rejected"}>{result.label}</div><p>Model: {result.model} · k={result.k}</p><p>Approval-labeled neighbor fraction: {(result.approval_neighbor_fraction*100).toFixed(1)}%</p><small>{result.notice}</small></>:<p className="muted">Complete the form and submit to see the model output.</p>}<hr/><p>Labels: Y → Approved, N → Rejected. The output reflects historical dataset labels, not a bank decision.</p></section></div>
 <footer>LoanLens · Waqas Khan Niazi · Next.js + FastAPI + scikit-learn</footer></main>;
}
