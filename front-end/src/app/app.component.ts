import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'front-end';
  isVisible = false;

  getLemma() {
    console.log('Lemma');
    this.isVisible = !this.isVisible;
  }
}
