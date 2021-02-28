from django.shortcuts import render
from .socipolicy import *
import datetime

def home(request):
    if request.method == 'POST':
        print(request.POST)
        forms = request.POST
        user = forms.get("user", "@CDCgov")
        dataset = forms.get("dataset", "mask")
        from_date = forms.get("from", "2020-10-09")
        target = forms.get("target", "2021-03-01")
        location = forms.get("location", "us")
        trends = forms.get("trends", -1)
        if user:
            baseline, tweeted, sample_size, graph_html = socipolicy(user, dataset, from_date, target, location, trends)

            baseline_output = ""
            tweeted_output = ""
            target = datetime.datetime.strptime(target, '%Y-%m-%d').strftime('%m/%d/%y')
            if dataset.lower() == 'vaccine':
                baseline_output = "Predicted baseline vaccine acceptance likelihood in {} on {}: {}%".format(location, target, round(baseline[0], 3))
                tweeted_output = "Predicted vaccine acceptance likelihood in {} on {} if a Tweet was made by {}: {}%".format(location, target, user, round(tweeted[0], 3))
            else:
                baseline_output = "Predicted baseline mask wearing likelihood in {} on {}: {}%".format(location, target, round(baseline[0], 3))
                tweeted_output = "Predicted mask wearing likelihood in {} on {} if a Tweet was made by {}: {}%".format(location, target, user, round(tweeted[0], 3))

            baseline_increase = baseline * sample_size / 100
            tweeted_increase = tweeted * sample_size / 100
            change = tweeted_increase - baseline_increase
            increase = "more"
            if change < 0:
                increase = "less"
            
            increase_output = ""
            if dataset.lower() == 'vaccine':
                increase_output = "With a average sample size of {}, Tweeting may cause {} {} people to accept vaccines.".format(round(sample_size), abs(int(change)), increase)
            else:
                increase_output = "With a average sample size of {}, Tweeting may cause {} {} people to wear masks.".format(round(sample_size), abs(int(change)), increase)


            return render(request, 'home.html', {"results": "Results",
                                                "baseline_output": baseline_output,
                                                "tweeted_output": tweeted_output,
                                                "increase_output": increase_output,
                                                "graph": graph_html
                                                })
        else:
            return render(request, 'home.html')
    else:
        return render(request, 'home.html')