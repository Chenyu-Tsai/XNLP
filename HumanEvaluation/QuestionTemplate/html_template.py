from dataclasses import dataclass
import xml.etree.ElementTree as ET

@dataclass
class html_template:
    question_file = """
<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
<HTMLContent><![CDATA[
<!DOCTYPE html>

]]>
</HTMLContent>
<FrameHeight>0</FrameHeight>
</HTMLQuestion>"""


    def generate_span_t1():
        file = "RTE5_test.xml"
        mapping = {'ENTAILMENT': 'Entailment', 'UNKNOWN': 'Neutral', 'CONTRADICTION': 'Contradiction'}
        root = ET.parse(file).getroot()
        premise = []
        hypothesis = []
        relation = []


        for type_tag in root.findall('pair'):

            e = type_tag.get('entailment')
            t = type_tag.find('t').text
            h = type_tag.find('h').text

            premise.append(t)
            hypothesis.append(h)
            relation.append(mapping[e])
        for i in range(int(len(premise)/20)):
            interval = (i + 1) * 20
            premise_section = premise[interval-20:interval]
            hypothesis_section = hypothesis[interval-20:interval]
            relation_section = relation[interval-20:interval]
            text = ""
            style = """<style>
  dt{
    font-size: medium;
  }

  dd{
    margin-bottom: 1rem; 
    margin-left: 0;
  }
  
  .card-text{
    margin-bottom: 2rem;
  }

  h5{
    font-weight: bold;
  }
  .tablink{
    font-weight: 600;
  }

  .btn-lg{
    font-size: 2rem;
    line-height: 4rem;
  }
  h2{
    margin-top: 1rem;
  }
  p{
    font-size: 1rem;
  }
</style>"""
            for idx in range(20):
                span = f"""  <div class="card shadow  mb-3 border-info text-center">
    <div class="card-header">Question {idx+1}</div>
    <div class="card-body">
      <h5 class="card-title">Premise</h5>
      <p class="card-text">{premise_section[idx]}</p>
      <h5 class="card-title">Hypoythesis</h5>
      <p class="card-text">{hypothesis_section[idx]}</p>
      <h5 class="card-title">Relation</h5>
      <p class="card-text">{relation_section[idx]}</p>
      <h5 class="card-title">Segment (Your response)</h5>
      <div class="form-group">
        <label for="question1"></label>
        <textarea type='text' class="form-control" name="{interval-20+1+idx}" id="{interval-20+1+idx}" rows="3"></textarea>
      </div>
      </div>
  </div>\n"""
                text += span
            text = f"""<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
<HTMLContent><![CDATA[
<!DOCTYPE html>
<html>
<head>
<meta http-equiv='Content-Type' content='text/html; charset=UTF-8'/>
<script type='text/javascript' src='https://s3.amazonaws.com/mturk-public/externalHIT_v1.js'></script>
<script src="https://sdk.amazonaws.com/js/aws-sdk-2.142.0.min.js"></script>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css" integrity="sha384-GJzZqFGwb1QTTN6wy59ffF1BuGJpLSa9DkKMp0DgiMDm4iYMj70gZWKYbI706tWS" crossorigin="anonymous">
<link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
<link rel="stylesheet" href="https://unpkg.com/bootstrap-material-design@4.1.1/dist/css/bootstrap-material-design.min.css" integrity="sha384-wXznGJNEXNG1NFsbm0ugrLFMQPWswR3lds2VeinahP8N0zJw9VWSopbjv2x7WCvX" crossorigin="anonymous">
{style}
</head>

<body>
<section class="container mb-3 mt-3">
  <fieldset class="shadow bg-white mb-3">
<div class="w3-container">
  <h2 class="text-info" style="font-weight: bolder">Instructions and Examples</h2>
  <div id="Instructions" class="w3-container ins w3-animate-opacity" style="display:block">
    <h2>Instructions</h2>
    <p>Read the premise, hypothesis and realation, select a segment from premise that can determine the relation between premise and hypothesis.</p>
    <ul>
      <!-- We are adding a template variable below which we will replace with an actual link in our code later -->
      <li>Read the premise, hypothesis and the relation.</li>
      <li>On the basis of your own judgement, select a "sequential" segment from premise that can determine the relation between premise and hypothesis.</li>
      <li>The length is not restricted, as long as the segment provides sufficient informations.</li>
      <li>Copy the segment from premise and paste it into the following box.</li>
      <li><strong>SPAM and non-sequential segments will be rejected.</strong></li>
      </ul>
    <p>The following section is the definition of the three realations, which comes from the RTE-5 task guideline. If you are familiar with NLI tasks such as SNLI, MNLI, RTE, you can skip the section below.</p>
    <p>We define textual entailment as a directional relationship between a pair of text fragments, which we call the "Premise" and the "Hypothesis". We say that: </p>
    <p><strong><i>Premise entails Hypothesis, if a human reading premise would infer that hypothesis is most likely true. </i></strong></p>
    <p>For example, given assumed common background knowledge of the business news domain and the following premise: </p>
    <p style="font-family: Times New Roman; font-size: large"><strong><i>Premise1 Internet media company Yahoo Inc. announced Monday it is buying Overture Services Inc. in a $1.63-billion (U.S.) cash-and-stock deal that will bolster its on-line search capabilities.</i></strong></p>
    <p>the following hypothesis is entailed:</p>
    <ul>
      <li style="font-family: Times New Roman; font-size: large"><p><i>H1.1 Yahoo boutght Overture</i></p></li>
      <li style="font-family: Times New Roman; font-size: large"><p><i>H1.2 Overture acquired by Yahoo</i></p></li>
      <li style="font-family: Times New Roman; font-size: large"><p><i>H1.3 Overture was bought</i></p></li>
      <li style="font-family: Times New Roman; font-size: large"><p><i>H1.4 Yahoo is an internet company</i></p></li>
    </ul>
    <p>For example, the following hypothesis are contradicted by Premise above:</p>
    <ul>
      <li style="font-family: Times New Roman; font-size: large"><p><i>H1.5 Overture bought Yahoo</i></p></li>
      <li style="font-family: Times New Roman; font-size: large"><p><i>H1.6 Yahoo sold Overture</i></p></li>
    </ul>
    <p>While the following ones cannot be judged on the basis of Premise above:</p>
    <ul>
      <li style="font-family: Times New Roman; font-size: large"><p><i>H1.7 Yahoo manufatures cars</i></p></li>
      <li style="font-family: Times New Roman; font-size: large"><p><i>H1.8 Overture shareholders will receive $4.75 cash and 0.6108 Yahoo stock for each of their shares.</i></p></li>
    </ul>
  </div>
  
</div>
</fieldset>
<fieldset class="shadow bg-white mb-3">
  <div id="Entailment" class="w3-container ins w3-animate-opacity">
    <h2>Entailment Example</h2>
    <p>Here is a pair of premise and hypothesis with entailment relation. The following is the Segment you need to respond, which we also emphasize in the premise.</p>
    <ul>
      <dt style="font-size: medium;">Premise</dt>
      <dd style='margin-bottom: 1rem; margin-left: 0;'>CNN) -- Malawians are rallying behind <strong class="w3-text-deep-orange">Madonna as she awaits a ruling Friday on whether she can adopt a girl from the southern African nation. The pop star, who has three children</strong>, adopted a son from Malawi in 2006. She is seeking to adopt Chifundo "Mercy" James, 4. "Ninety-nine percent of the people calling in are saying, 'let her take the baby,' " said Marilyn Segula, a presenter at Capital FM, which broadcasts in at least five cities, including the capital, Lilongwe.</t>
      <h>Madonna has three children.</dd>
      <dt>Hypothesis</dt>
      <dd>Madonna has three children.</dd>
      <dt>Relation</dt>
      <dd>Entailment</dd>
      <dt>Segment (Your response)</dt>
      <dd>Madonna as she awaits a ruling Friday on whether she can adopt a girl from the southern African nation. The pop star, who has three children</dd>
    </ul>
  </div>
</fieldset>
<fieldset class="shadow bg-white mb-3">
  <div id="Neutral" class="w3-container ins w3-animate-opacity">
    <h2>Neutral Example</h2>
    <p>Here is a pair of premise and hypothesis with neutral relation. The following is the Segment you need to respond, which we also emphasize in the premise.</p> 
    <ul>
      <dt>Premise</dt>
      <dd>Concerns have been raised that potential leads in the hunt for missing York woman <strong class="w3-text-deep-orange">Claudia Lawrence are not being followed up quickly enough. It comes after hotel staff in Malton contacted police</strong> after a stranger in the bar expressed satisfaction when asked if the chef was still missing. The incident happened more than three weeks ago and staff said detectives had not yet been in touch. Police said leads were being assessed in a methodical and structured way.</dd>
      <dt>Hypothesis</dt>
      <dd>Claudia Lawrence resides in Malton.</dd>
      <dt>Relation</dt>
      <dd>Neutral</dd>
      <dt>Segment (Your response)</dt>
      <dd>Claudia Lawrence are not being followed up quickly enough. It comes after hotel staff in Malton contacted police</dd>
    </ul>
  </div>
</fieldset>
<fieldset class="shadow bg-white mb-3">
  <div id="Contradiction" class="w3-container ins w3-animate-opacity">
    <h2>Contradiction Example</h2>
    <p>Here is a pair of premise and hypothesis with Contradiction relation. The following is the Segment you need to respond, which we also emphasize in the premise.</p> 
    <ul>
      <dt>Premise</dt>
      <dd>But an O2 insider said there had been problems with a companies trying to sell the pass codes. Some people were "bound to have been turned away" because of fraudulent tickets. <strong class="w3-text-deep-orange">Led Zeppelin, formed in 1968</strong>, were one of the most influential bands of the 1970s with songs such as Whole Lotta Love and Stairway To Heaven. They split in 1980 after the death of the drummer John Bonham. Kenneth Donnell, 25, who was not born in 1980, spent 83,000 on two tickets in a BBC Children in Need auction.</dd>
      <dt>Hypothesis</dt>
      <dd>The band Led Zeppelin was formed in 1980.</dd>
      <dt>Relation</dt>
      <dd>Contradiction</dd>
      <dt>Segment (Your response)</dt>
      <dd>Led Zeppelin, formed in 1968</dd>
    </ul>
  </div>
</fieldset>
</section>

<!-- Questions Section -->
<section class="container">
  <div class="alert alert-danger text-center shadow" role="alert" style="font-weight: bold; font-size: large">
  All answers will be manually checked, SPAM and blank answers will be definitely rejected.
  </div>  
  <form name='mturk_form' method='post' id='mturk_form' action='https://www.mturk.com/mturk/externalSubmit'><input type='hidden' value='' name='assignmentId' id='assignmentId'/>
  {text}
  <button type="submit" class="btn btn-info btn-raised btn-lg btn-block" id='submitButton'>Submit</button>
  
</form>
</section>
<script language='Javascript'>turkSetAssignmentID();</script>

     

</body>
</html>

<script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.6/umd/popper.min.js" integrity="sha384-wHAiFfRlMFy6i5SRaxvfOCifBUQy1xHdJ/yoi7FRNXMRBu5WHdZYu1hA6ZOblgut" crossorigin="anonymous"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.2.1/js/bootstrap.min.js" integrity="sha384-B0UglyR+jN6CkvvICOB2joaf5I4l3gm9GU6Hc1og6Ls7i6U/mkkaduKaBhlAXv9k" crossorigin="anonymous"></script>
]]>
</HTMLContent>
<FrameHeight>0</FrameHeight>
</HTMLQuestion>"""
            file_name = f'questions{i+1}.xml'
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(text)
                f.close

def main():
    html_template.generate_span_t1()

if __name__ == '__main__':
    main()