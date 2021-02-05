<<<<<<< HEAD
<script src="https://d3js.org/d3.v5.js"></script>
import * as d3 from "d3";

$(document).ready(function() {
	$('.nav-btn').on('click', function(event) {
		event.preventDefault();
		/* Act on the event */
		$('.sidebar').slideToggle('fast');

		window.onresize = function(){
			if ($(window).width() >= 768) {
				$('.sidebar').show();
			} else {
				$('.sidebar').hide();
			}
		};
	});
});

/*
Trid tkun hekk:
https://moviegalaxies.com/movies/view/773/star-wars-episode-iv-a-new-hope/#

Word Cloud by d3:
https://observablehq.com/@d3/word-cloud

Force-Directed Graph by d3:
https://observablehq.com/@d3/force-directed-graph
*/
=======
$(document).ready(function() {
	$('.nav-btn').on('click', function(event) {
		event.preventDefault();
		/* Act on the event */
		$('.sidebar').slideToggle('fast');

		window.onresize = function(){
			if ($(window).width() >= 768) {
				$('.sidebar').show();
			} else {
				$('.sidebar').hide();
			}
		};
	});
});
>>>>>>> 208b37f0e2cb95a83c545038e815ce8459172c95
